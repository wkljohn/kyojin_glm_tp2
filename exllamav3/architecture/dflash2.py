from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F

from ..model.config import no_default
from ..modules import RMSNorm, TransformerBlock, Linear
from ..modules.module import Module
from .dflash import DFlashConfig, DFlashModel

# DFlash 2 drafter (incoai GLM-5.3-Flash-DFlash2, z-lab dflash v2). Differences from DFlash (dflash.py):
#  - every decoder layer wraps attention and MLP in a grouped two-tap dynamic convolution over the
#    drafted block (base kernel + per-token, per-group coefficients projected from the layer input)
#  - drafting keeps the top-k candidates per block position and a low-rank selector scores
#    (predecessor, candidate) edges; a greedy walk from the anchor token picks one path
#  - target states are taken at the raw target_layer_ids ("output of layer i": vLLM captures them
#    at ids + 1 in its capture-point indexing, see dflash_laguna.py), and the drafter ships neither
#    embeddings nor a head: both come from the attached target
# Reference: vllm model_executor/models/qwen3_dflash2.py and v1/spec_decode/dflash.py (_sample_draft_tokens).


class DFlash2Config(DFlashConfig):
    arch_string = "DFlash2DraftModel"
    tap_shift = 0

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": DFlash2Model}, **kwargs)
        self.conv_taps = self.read_cfg(int, "dflash_config->conv_kernel_size", no_default)
        self.conv_group_size = self.read_cfg(int, "dflash_config->conv_group_size", no_default)
        self.selector_rank = self.read_cfg(int, "dflash_config->selector_rank", no_default)
        self.selector_top_k = self.read_cfg(int, "dflash_config->selector_top_k", no_default)
        self.input_embedding_scale = self.read_cfg(float, "dflash_config->input_embedding_scale", 1.0)
        self.output_multiplier = self.read_cfg(float, "dflash_config->output_multiplier", 1.0)
        self.final_logit_softcapping = self.read_cfg(float, "dflash_config->final_logit_softcapping", 0.0) or 0.0
        assert self.hidden_size % self.conv_group_size == 0


class DFlash2GroupedConv(Module):
    """Grouped dynamic convolution over the query block: out[t] = sum_tap c_tap[t] * x[t - tap], with
    c_tap = base[side, tap] + delta[t, side, tap, group] (broadcast over the group's channels) and taps that
    reach before the block start dropped. prepare() convolves the sublayer input (side 0) and returns the
    side-1 coefficients for finish() on its output."""

    def __init__(self, config, key: str, hidden_size: int, taps: int, group_size: int, block_size: int):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.block_size = block_size
        self.base_kernel = None
        self.kernel_projection = Linear(
            config = config,
            key = f"{key}.kernel_projection",
            in_features = hidden_size,
            out_features = 2 * taps * self.num_groups,
            qmap = None,
            out_dtype = torch.float,
            pad_to = 1,
        )
        self.register_submodule(self.kernel_projection)

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        bk = self.config.stc.get_tensor(f"{self.key}.base_kernel", device, allow_bf16 = True, no_defer = True)
        self.base_kernel = bk.float().view(2, self.taps, self.num_groups, self.group_size)

    @override
    def unload(self):
        super().unload()
        self.base_kernel = None

    @override
    def can_defer_load(self):
        return False

    @override
    def get_tensors(self):
        return {f"{self.key}.base_kernel": self.base_kernel} if self.base_kernel is not None else {}

    def _convolve(self, x: torch.Tensor, delta: torch.Tensor, side: int) -> torch.Tensor:
        # x (bsz, seq, H) any float dtype, delta (bsz, seq, taps, G) fp32 -> (bsz, seq, H) fp32
        bsz, seq, _ = x.shape
        blocks = x.float().view(bsz, seq, self.num_groups, self.group_size)
        coef = self.base_kernel[side].view(1, 1, self.taps, self.num_groups, self.group_size) + delta.unsqueeze(-1)
        out = coef[:, :, 0] * blocks
        pos = torch.arange(seq, device = x.device) % self.block_size
        for tap in range(1, self.taps):
            shifted = F.pad(blocks[:, :-tap], (0, 0, 0, 0, tap, 0))
            out += coef[:, :, tap] * shifted * (pos >= tap).view(1, -1, 1, 1)
        return out.view(bsz, seq, self.hidden_size)

    def prepare(self, x: torch.Tensor, params: dict):
        bsz, seq, _ = x.shape
        c = self.kernel_projection.forward(x.half(), params).view(bsz, seq, 2, self.taps, self.num_groups)
        return self._convolve(x, c[:, :, 0], 0).half(), c[:, :, 1]

    def finish(self, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self._convolve(y, c, 1)

    def optimizer_targets(self):
        return []

    def forward(self, x, params, out_dtype = None):
        raise NotImplementedError()


class DFlash2Block(TransformerBlock):
    """Pre-norm residual block with the attention and MLP sublayers wrapped in grouped convolutions."""

    def __init__(self, attn_conv: DFlash2GroupedConv, mlp_conv: DFlash2GroupedConv, **kwargs):
        super().__init__(**kwargs)
        self.attn_conv = attn_conv
        self.mlp_conv = mlp_conv
        self.register_submodule(attn_conv)
        self.register_submodule(mlp_conv)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        x = x.float()
        y = self.attn_norm.forward(x, params, out_dtype = torch.half)
        y, c = self.attn_conv.prepare(y, params)
        y = self.attn.forward(y, params)
        x = x + self.attn_conv.finish(y, c)
        y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
        y, c = self.mlp_conv.prepare(y, params)
        y = self.mlp.forward(y, params)
        x = x + self.mlp_conv.finish(y, c)
        return x.to(out_dtype) if out_dtype is not None else x


class DFlash2Selector(Module):
    """Candidate-path selector weights. Sits after the final norm in the module list as an identity so it loads
    with the model; sample_from_state() uses it."""

    def __init__(self, config: DFlash2Config, key: str):
        super().__init__(config, key, None)
        self.hidden_projection = Linear(
            config = config,
            key = f"{key}.hidden_projection",
            in_features = config.hidden_size,
            out_features = config.selector_rank,
            qmap = None,
            out_dtype = torch.float,
            pad_to = 1,
        )
        self.register_submodule(self.hidden_projection)
        self.predecessor = None
        self.successor = None

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.predecessor = stc.get_tensor(f"{self.key}.predecessor_codebook", device, float2half = True, no_defer = True)
        self.successor = stc.get_tensor(f"{self.key}.successor_codebook", device, float2half = True, no_defer = True)

    @override
    def unload(self):
        super().unload()
        self.predecessor = self.successor = None

    @override
    def can_defer_load(self):
        return False

    @override
    def get_tensors(self):
        if self.predecessor is None:
            return {}
        return {f"{self.key}.predecessor_codebook": self.predecessor,
                f"{self.key}.successor_codebook": self.successor}

    @override
    def weights_numel(self):
        return super().weights_numel() + 2 * self.config.vocab_size * self.config.selector_rank

    def optimizer_targets(self):
        return []

    def forward(self, x, params, out_dtype = None):
        return x

    def select(self, cand: torch.Tensor, unary: torch.Tensor, hidden: torch.Tensor, anchor: torch.Tensor):
        """cand/unary (b, L, k), hidden (b, L, H) fp16, anchor (b,) -> greedy path (b, L) token ids."""
        b, L, k = cand.shape
        h = self.hidden_projection.forward(hidden, {})                                # (b, L, r) fp32
        prev = torch.cat((anchor.view(b, 1, 1).expand(b, 1, k), cand[:, :-1]), dim = 1)  # (b, L, k)
        pred = self.predecessor[prev].float()                                        # (b, L, k, r)
        succ = self.successor[cand].float()                                          # (b, L, k, r)
        scores = unary.unsqueeze(2) + torch.einsum("blpr,blcr->blpc", pred * h.unsqueeze(2), succ)
        rows = torch.arange(b, device = cand.device)
        p = torch.zeros(b, dtype = torch.long, device = cand.device)
        path, conf = [], []
        for step in range(L):
            s = scores[rows, step, p]                                                # (b, k)
            sel = s.argmax(dim = -1)
            path.append(cand[rows, step, sel])
            conf.append(s[rows, sel])
            p = sel
        return torch.stack(path, dim = 1), torch.stack(conf, dim = 1)


class DFlash2Model(DFlashModel):
    config_class = DFlash2Config

    def __init__(self, config: DFlash2Config, **kwargs):
        super().__init__(config, **kwargs)
        self.input_layer.input_embedding_scale = config.input_embedding_scale

        # Cross-host expert parallel: the drafter runs replicated on every rank; its MLPs (most of its weight bytes)
        # are split by intermediate channels like the target's dense layers, reduced through GatedMLP.tp_reduce
        import os
        from ..model.ep import ep_state
        from ..modules.mlp import ep_slice_gated_mlp
        ep = ep_state()
        split_mlp = ep is not None and os.environ.get("KYOJIN_EP_DRAFT_SPLIT", "1") != "0"

        # Re-wrap the decoder blocks built by DFlashModel with the grouped convolutions
        for i in range(self.first_block_idx, self.first_block_idx + config.num_hidden_layers):
            blk = self.modules[i]
            idx = blk.layer_idx
            if split_mlp:
                sliced = ep_slice_gated_mlp(blk.mlp, ep, config)
                if sliced is not None:
                    blk.mlp = sliced
                    sliced.ep = ep
                    sliced.tp_reduce = True
            mk = lambda tag: DFlash2GroupedConv(
                config, f"layers.{idx}.{tag}", config.hidden_size, config.conv_taps,
                config.conv_group_size, config.block_size,
            )
            self.modules[i] = DFlash2Block(
                attn_conv = mk("attention_conv"),
                mlp_conv = mk("mlp_conv"),
                config = config,
                key = blk.key,
                layer_idx = idx,
                attn_norm = blk.attn_norm,
                attn = blk.attn,
                mlp_norm = blk.mlp_norm,
                mlp = blk.mlp,
            )

        self.selector = DFlash2Selector(config, "candidate_selector")
        self.modules += [self.selector]
        self.caps.update({"dflash2_draft": True})

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # Anchor token of each row (the last accepted token), the selector's first predecessor
        params["dflash2_anchor"] = input_ids[:, -1].clone()
        return super().prepare_inputs(input_ids, params)

    @override
    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        target = self.attached_model()
        assert not target.loaded_tp, "DFlash2 with the in-host TP loader is not supported"
        cfg = self.config
        lm = target.modules[target.logit_layer_idx]
        hidden = state[:, 1:, :]                                    # mask positions; position 0 is the anchor
        logits = lm.forward(lm.prepare_for_device(hidden, params), params)[..., :target.config.vocab_size].float()
        if cfg.output_multiplier != 1.0:
            logits = logits * cfg.output_multiplier
        if cfg.final_logit_softcapping:
            logits = torch.tanh(logits / cfg.final_logit_softcapping) * cfg.final_logit_softcapping
        unary, cand = logits.topk(cfg.selector_top_k, dim = -1)
        anchor = params["dflash2_anchor"].to(state.device)
        path, conf = self.selector.select(cand, unary, hidden.half(), anchor)
        if params.get("export_draft_conf"):
            # Per-position confidence for the generator's calibrated draft truncation: the selector score of the
            # chosen edge (column 0 is the anchor position the generator drops)
            params["draft_conf"] = torch.cat((conf[:, :1], conf), dim = 1)
        # The generator drops column 0 (the anchor position)
        return torch.cat((anchor.view(-1, 1), path), dim = 1)
