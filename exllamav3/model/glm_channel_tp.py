"""Rank-local GLM MoE channel loading using the original packed EXL3 tensors."""
from __future__ import annotations

import torch

from ..modules.block_sparse_mlp import BlockSparseMLP
from ..modules.quant.exl3 import LinearEXL3


class _CPUSlices:
    def __init__(self, device):
        self.device = torch.device(device)

    def recv(self, tensor, *, cuda=False, slice_dim=None, first=None, last=None):
        if tensor is None:
            return None
        if tensor.device.type != "cpu":
            raise ValueError("Slice packed source tensors on CPU before GPU upload")
        if slice_dim is not None:
            index = [slice(None)] * tensor.ndim
            index[slice_dim] = slice(first, last)
            tensor = tensor[tuple(index)]
        return tensor.contiguous().to(self.device if cuda else "cpu")


def load_channel_linear(linear, device):
    """Load a configured channel slice, with at most one full source projection on CPU."""
    split_out, first, last = linear.rank_channel_split
    source_in, source_out, unpadded_in, unpadded_out = linear.rank_channel_source_dims
    if first % 128 or last % 128 or not 0 <= first < last <= (source_out if split_out else source_in):
        raise ValueError(f"{linear.key}: EXL3 splits require whole 128-channel blocks")
    stc = linear.config.stc
    key = linear.key
    context = {"consumer": _CPUSlices(device), "device": torch.device(device)}
    exported = {"in_features": source_in, "out_features": source_out, "out_dtype": linear.out_dtype}
    if linear.is_exl3_storage(key):
        # This selected pack stores unfolded FP16 sign vectors and mcg markers.
        # Use the existing explicit splitter; ordinary Linear.load_exl3 ignores
        # first/full fields and cannot materialize a rank-local shard by itself.
        for name in ("suh", "svh", "trellis", "bias", "mcg", "mul1"):
            exported[name] = stc.get_tensor(key + "." + name, "cpu", optional=name in ("bias", "mcg", "mul1"), no_defer=True)
        inner = LinearEXL3.tp_import_split(context, exported, {}, linear.rank_channel_split)
        inner.config = linear.config
        inner.key = linear.key
    else:
        raise ValueError(f"{key}: channel sharding requires unfused EXL3 routed weights")
    linear.inner = inner
    linear.device = torch.device(device)
    linear.quant_type = inner.quant_type
    linear._adopt_inner_dims(
        {"kwargs": {"in_features": source_in, "out_features": source_out},
         "unpadded": (unpadded_in, unpadded_out)},
        out_first=first if split_out else None, in_first=first if not split_out else None,
    )


def configure_glm_channel_tp(model, backend):
    """Shard every routed expert's intermediate; replicate ordinary attention.

    Call once before loading. The shared expert belongs to global rank zero;
    the parent MoE reduces it together with routed partials exactly once.
    This does not invoke the local multiprocessing/shared-memory TP launcher.
    """
    if model.config.architecture != "Glm5NextForConditionalGeneration" or backend.world_size != 2:
        raise ValueError("This loader requires GLM-5.3 and two global ranks")
    if getattr(model, "rank_collectives", None) is not None:
        raise ValueError("Rank-local channel sharding is already configured")
    ip = model.config.infer_params
    if ip.moe_cpu_offload or ip.moe_cpu_split:
        raise ValueError("CPU expert offload and distributed channel sharding cannot be combined")
    modules = [m for m in model if isinstance(m, BlockSparseMLP)]
    if not modules:
        raise ValueError("The model contains no routed expert layers")
    for module in modules:
        if module.device is not None or module.intermediate_size % 256:
            raise ValueError("Configure unloaded experts with a 256-aligned intermediate")
        if module.num_local_experts != module.num_experts or module.tp_mode is not None:
            raise ValueError("The source must contain every complete expert")
        if module.routed_post_norm is not None or module.shared_experts_post_norm is not None:
            raise ValueError("Channel sharding does not support expert post normalization")
        for linear in module.gates + module.ups + module.downs:
            if linear.fkey is not None or not linear.is_exl3_storage(linear.key):
                raise ValueError(f"{linear.key}: channel sharding requires unfused EXL3 routed weights")
    for module in modules:
        width = module.intermediate_size // backend.world_size
        first, last = backend.rank * width, (backend.rank + 1) * width
        for linear in module.gates + module.ups:
            linear.rank_channel_source_dims = (
                linear.in_features, linear.out_features,
                linear.in_features_unpadded, linear.out_features_unpadded,
            )
            linear.rank_channel_split = (True, first, last)
        for linear in module.downs:
            linear.rank_channel_source_dims = (
                linear.in_features, linear.out_features,
                linear.in_features_unpadded, linear.out_features_unpadded,
            )
            linear.rank_channel_split = (False, first, last)
        module.intermediate_size = width
        module.intermediate_size_padded = width
        module.tp_mode = "channels"
        module.tp_reduce = True
        # The backend interprets this legacy field as a GLOBAL source rank.
        module.routing_device = 0
        module.routing_first = 0
        module.routing_last = module.num_experts
        if backend.rank != 0 and module.shared_experts is not None:
            module.modules.remove(module.shared_experts)
            module.shared_experts = None
    model.rank_collectives = backend
    return len(modules)
