"""
KYOJIN_EP_SHARED_DYN=1: the shared expert as a per-call load balancer between the two EP ranks.

Each MoE layer's routed work splits unevenly between the ranks from one decode round to the next (the rank owning
more of the picked experts reads more expert weights), and the faster rank idles at the layer's all-reduce. Both ranks
know the routing, so both can split the shared expert's intermediate channels to even out the predicted loads. The
shared expert's Q8 weights are replicated (full width on both ranks, ~25 MB per layer) and ep_shared_range computes
this rank's channel range on the device, so decode needs no host sync. Prefill keeps the static share split
(KYOJIN_EP_SHARE_DENSE), on views of the transient dequantized weight.

KYOJIN_EP_SHDYN = "sb,w1,b1": cost of one 128-channel shared block in routed-expert units, rank 1's cost per unit
relative to rank 0, rank 1's fixed extra time in the MoE segment in units. Tuned on the verify probe.
"""
from __future__ import annotations
import os
import torch
from ..ext import exllamav3_ext as ext

_SHDYN = [float(v) for v in os.environ.get("KYOJIN_EP_SHDYN", "0.15,1.13,0").split(",")]
MAX_ROWS = 8


class EPSharedDyn:

    def __init__(self, se, ep, num_local_experts: int, num_experts: int):
        self.se = se
        self.ep = ep
        self.num_local = num_local_experts
        self.num_experts = num_experts
        self.I = se.intermediate_size
        assert self.I % 128 == 0
        self.nb = self.I // 128
        a, n = ep.split(self.I, 128)
        self.static = (a, a + n)
        self.rng = {}

    def _lin(self, l):
        inner = getattr(l, "inner", l)
        return inner.qweight, inner.qscale

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        sel = params.get("ep_shared_sel")
        gate, up, down = self.se.gates[0], self.se.ups[0], self.se.downs[0]
        gq, gs = self._lin(gate)
        uq, us = self._lin(up)
        dq, ds = self._lin(down)
        H = x.shape[-1]
        if rows <= MAX_ROWS and sel is not None and gq is not None and x.dtype == torch.half:
            xv = x.reshape(rows, H)
            rng = self.rng.get(x.device)
            if rng is None:
                rng = self.rng[x.device] = torch.zeros(2, dtype = torch.int, device = x.device)
            sb, w1, b1 = _SHDYN
            ext.ep_shared_range(sel.reshape(-1), self.num_local, self.num_experts, self.ep.rank, self.nb, sb, w1, b1, rng)
            g = torch.empty((rows, self.I), dtype = torch.half, device = x.device)
            u = torch.empty_like(g)
            ext.hgemm_q8_range(xv, gq, gs, g, rng, 0)
            ext.hgemm_q8_range(xv, uq, us, u, rng, 0)
            a = torch.empty_like(g)
            self.se.activation_fn_call(g, u, a, self.se.act_limit)
            y = torch.empty((rows, H), dtype = torch.float, device = x.device)
            ext.hgemm_q8_range(a, dq, ds, y, rng, 1)
            return y.view(x.shape[:-1] + (H,))
        # Prefill: this rank's static channel slice of the dequantized weights
        lo, hi = self.static
        xv = x.reshape(rows, H)
        wg = gate.inner._dequant_prefill() if gq is not None else gate.inner.weight
        wu = up.inner._dequant_prefill() if uq is not None else up.inner.weight
        g = xv @ wg[:, lo:hi]
        u = xv @ wu[:, lo:hi]
        a = torch.empty_like(u, dtype = torch.half)
        self.se.activation_fn_call(g, u, a, self.se.act_limit)
        wd = down.inner._dequant_prefill() if dq is not None else down.inner.weight
        half = bool(params.get("nc2_down_half"))
        y = a @ wd[lo:hi]
        if not half:
            y = y.float()
        return y.view(x.shape[:-1] + (H,))
