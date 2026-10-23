#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
);

#if defined(USE_ROCM)
void skinny_set_onepass(int64_t on);
#if defined(USE_ROCM)
void skinny_cat(at::Tensor a, at::Tensor b, std::vector<at::Tensor> outs);
void skinny_cat_w(at::Tensor a, std::vector<at::Tensor> ws, std::vector<at::Tensor> outs);
void q8_dequant(at::Tensor qw, at::Tensor qs, at::Tensor out);
void hgemm_q8_gr(at::Tensor a, at::Tensor qw, at::Tensor qs, at::Tensor c, Graph* graph);
void hgemm_q8(at::Tensor a, at::Tensor qw, at::Tensor qs, at::Tensor c);
void hgemm_q8_range(at::Tensor a, at::Tensor qw, at::Tensor qs, at::Tensor c, at::Tensor rng, int64_t mode);
void ep_shared_range(at::Tensor sel, int64_t n_local, int64_t n_experts, int64_t rank, int64_t nb, double sb, double w1, double b1, at::Tensor rng);
void dflash2_conv(at::Tensor x, at::Tensor base, at::Tensor delta, at::Tensor out, int64_t block);
void dflash2_walk(at::Tensor scores, at::Tensor sel);
#endif
#endif

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
);
void hgemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
);

// fp16-accumulator tensor-core GEMM with fp32 accumulation across K (hgemm_f16acc.cu): full
// rate on GeForce, where the fp32-accumulator MMA runs at half rate. hgemm_f16acc_try runs it
// when the device gains (rate probe, EXL3_HGEMM_F16ACC=0/1 overrides) and the shape is covered
// (K % 64, N % 128, aligned packed input rows and nonoverlapping output rows/batches).
// Shape/batch heuristics decide whether it pays; hgemm_recon adds the cuBLAS fallback.
bool hgemm_f16acc_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c);
void hgemm_f16acc(at::Tensor a, at::Tensor b, at::Tensor c);
int hgemm_f16acc_status(int device);
void hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c);
