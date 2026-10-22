#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include "pad_pitch.h"
#include <limits>
#include <cstdlib>
#include <type_traits>

/*

Row-major matmul using cuBLAS, a @ b -> c
- if c is float16, operation is float16 @ float16 -> float16 (float16 accumulate)
- if c is float32, operation is float16 @ float16 -> float32 (float32 accumulate)
*/

using bfloat16 = __nv_bfloat16;

// row stride (elements) of the activation operand. Contiguous -> k (unchanged behaviour).
// A row-padded view (stride > k, columns contiguous, leading dims collapsible) is accepted so a producer can leave
// a non-power-of-two row pitch: K=6144 fp16 rows are 12288 B apart and the GEMM reads them at ~half rate.
// Returns false if the view cannot be expressed as one (m, k) matrix with a single row pitch or if the
// storage cannot hold (m-1)*lda + k elements (the bound that keeps the library from reading outside the buffer).
static bool a_row_pitch(const at::Tensor& a, int64_t& lda)
{
    const int dim = a.dim();
    if (dim < 1) return false;
    const int64_t k = a.size(-1);
    if (k < 1) return false;
    if (dim == 1) { lda = k; return true; }
    if (a.stride(-1) != 1) return false;
    int64_t m = a.size(-2);
    for (int i = dim - 3; i >= 0; --i)
    {
        if (a.size(i + 1) != 1 && a.stride(i) != a.stride(i + 1) * a.size(i + 1)) return false;
        m *= a.size(i);
    }
    lda = m == 1 ? k : a.stride(-2);
    if (lda < k || lda > std::numeric_limits<int>::max()) return false;
    return pad_view_fits(a.storage_offset(), m, lda, k, (int64_t) (a.storage().nbytes() / sizeof(at::Half)));
}

static void hgemm_gemmex_impl
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    cudaStream_t stream
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    bool output_fp32 = c.dtype() == at::kFloat;
    bool output_fp16 = c.dtype() == at::kHalf;

    TORCH_CHECK(output_fp32 || output_fp16, "c must be float32 or float16");

    // Check shapes of a,b,c are compatible
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(b, kHalf);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() >= 2, "c must have at least 2 dimensions");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    const half* a_ptr = (const half*) a.data_ptr();
    const half* b_ptr = (const half*) b.data_ptr();

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    int size_n = b.size(-1);
    int64_t a_pitch = 0;
    TORCH_CHECK(a_row_pitch(a, a_pitch), "hgemm: a must be contiguous or a row-padded view inside its storage");
    int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
    TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(), "c row stride is too large");

    // Set cuBLAS modes and workspace
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cudaDataType_t c_type = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
    auto r = cublasGemmEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, b_ptr, CUDA_R_16F, size_n,
                 a_ptr, CUDA_R_16F, (int) a_pitch,
        &beta_,  c.data_ptr(), c_type, (int) c_stride_m,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

#if defined(USE_ROCM)

/*
Tuned dense GEMM for ROCm prefill (EXL3_DENSE_GEMM_TUNE=1, read on every call so an in-process
A/B can toggle it). The default hipblas -> rocBLAS/hipBLASLt heuristic picks MT128x128x16 for
every large fp16 GEMM on gfx1151 (~24-27 TFLOPS); a rocBLAS solution chosen per shape by
timing runs the same GEMMs 20-35% faster. Per (n, k, m class, out dtype) key, every solution
rocblas_gemm_ex_get_solutions returns is screened on the live operands, the best 4 are re-timed
against the untuned hipblas path and must beat it by 3%; the winner (or 0 = untuned) is kept in
memory and appended to a cache file (keyed by rocBLAS version and arch), so a process tunes each shape once ever. Own rocblas handle with atomics disabled:
split-K solutions stay deterministic. Never tunes under stream capture; any failure falls back
to the untuned path.
*/

#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#include <algorithm>
#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <mutex>
#include <cstdio>
#include <sys/stat.h>

namespace dtune
{

constexpr int MIN_M = 256;             // rows below this: untuned path (decode, short tails)
constexpr int NUM_DEV = MAX_DEVICES;

typedef std::tuple<int, int, int, int, int> Key;   // arch-local: n, k, m class, ldc == n, out fp32
static std::map<Key, int> g_cache;     // solution index, 0 = default heuristic wins
static bool g_loaded = false;
static std::mutex g_mutex;
static rocblas_handle g_handle[NUM_DEV] = {};
static std::string g_tag;

static bool enabled()
{
    const char* e = getenv("EXL3_DENSE_GEMM_TUNE");
    return !e || *e != '0';   // on by default: winners are bit-exact with the untuned path
}

// m class: tuned winners at 2048 and 1792 rows coincide; round up to 256 so one key serves
// every chunk length in the class
static int m_class(int m) { return ((m + 255) / 256) * 256; }

static std::string cache_path()
{
    const char* p = getenv("EXL3_DENSE_GEMM_TUNE_FILE");
    if (p && *p) return p;
    const char* h = getenv("HOME");
    std::string d = std::string(h ? h : "/tmp") + "/.cache/exllamav3";
    mkdir((std::string(h ? h : "/tmp") + "/.cache").c_str(), 0755);
    mkdir(d.c_str(), 0755);
    return d + "/dense_gemm_tune.txt";
}

static void load(int device)
{
    if (g_loaded) return;
    g_loaded = true;
    char ver[256] = {};
    rocblas_get_version_string(ver, sizeof(ver));
    hipDeviceProp_t prop;
    cudaGetDeviceProperties(&prop, device);
    g_tag = std::string(ver) + "|" + prop.gcnArchName;
    for (auto& ch : g_tag) if (ch == ' ') ch = '_';
    g_tag += "|exact1";   // only bit-exact winners; older free-pick lines no longer match
    FILE* f = fopen(cache_path().c_str(), "r");
    if (!f) return;
    char tag[512]; int n, k, mc, ldn, f32, sol;
    while (fscanf(f, "%511s %d %d %d %d %d %d", tag, &n, &k, &mc, &ldn, &f32, &sol) == 7)
        if (g_tag == tag) g_cache[Key(n, k, mc, ldn, f32)] = sol;
    fclose(f);
}

static void store(const Key& key, int sol)
{
    FILE* f = fopen(cache_path().c_str(), "a");
    if (!f) return;
    fprintf(f, "%s %d %d %d %d %d %d\n", g_tag.c_str(), std::get<0>(key), std::get<1>(key),
            std::get<2>(key), std::get<3>(key), std::get<4>(key), sol);
    fclose(f);
}

static rocblas_handle handle(int device)
{
    if (!g_handle[device])
    {
        rocblas_create_handle(&g_handle[device]);
        rocblas_set_atomics_mode(g_handle[device], rocblas_atomics_not_allowed);
        rocblas_set_pointer_mode(g_handle[device], rocblas_pointer_mode_host);
    }
    return g_handle[device];
}

struct Call
{
    rocblas_handle h; int m, n, k, ldc, lda; const void* a; const void* b; void* c; rocblas_datatype ct;
    rocblas_status run(rocblas_gemm_algo algo, int sol) const
    {
        float alpha = 1.0f, beta = 0.0f;
        // column-major: C^T[n, m] = B^T[n, k] @ A^T[k, m], same mapping as hgemm_gemmex_impl
        return rocblas_gemm_ex(h, rocblas_operation_none, rocblas_operation_none, n, m, k,
                               &alpha, b, rocblas_datatype_f16_r, n, a, rocblas_datatype_f16_r, lda,
                               &beta, c, ct, ldc, c, ct, ldc, rocblas_datatype_f32_r,
                               algo, sol, rocblas_gemm_flags_none);
    }
};

template <typename F>
static float time_fn(F&& fn, cudaStream_t stream, int reps)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    if (!fn()) { cudaEventDestroy(e0); cudaEventDestroy(e1); return 1e30f; }
    cudaEventRecord(e0, stream);
    for (int i = 0; i < reps; ++i) fn();
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, e0, e1) != cudaSuccess) ms = 1e30f * reps;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / reps;
}

static float time_ms(const Call& call, rocblas_gemm_algo algo, int sol, cudaStream_t stream, int reps)
{
    return time_fn([&]{ return call.run(algo, sol) == rocblas_status_success; }, stream, reps);
}

// def_time(reps): times the untuned path (hipblas GemmEx, may route to hipBLASLt), the bar to beat
// same(): true if c now holds exactly the untuned path's output. Only bit-identical solutions
// qualify, so tuning never changes numerics (a free pick moved PPL +0.11% at m=4096).
template <typename T, typename S>
static int tune(const Call& call, cudaStream_t stream, T&& def_time, S&& same)
{
    float alpha = 1.0f, beta = 0.0f;
    rocblas_int count = 0;
    auto st = rocblas_gemm_ex_get_solutions(call.h, rocblas_operation_none, rocblas_operation_none,
        call.n, call.m, call.k, &alpha, call.b, rocblas_datatype_f16_r, call.n, call.a,
        rocblas_datatype_f16_r, call.k, &beta, call.c, call.ct, call.ldc, call.c, call.ct, call.ldc,
        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, rocblas_gemm_flags_none, nullptr, &count);
    if (st != rocblas_status_success || count <= 0) return 0;
    std::vector<rocblas_int> sols(count);
    rocblas_gemm_ex_get_solutions(call.h, rocblas_operation_none, rocblas_operation_none,
        call.n, call.m, call.k, &alpha, call.b, rocblas_datatype_f16_r, call.n, call.a,
        rocblas_datatype_f16_r, call.k, &beta, call.c, call.ct, call.ldc, call.c, call.ct, call.ldc,
        rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, rocblas_gemm_flags_none, sols.data(), &count);
    // screen every solution with one timed rep, then re-time the best 4 (and the default)
    std::vector<std::pair<float, int>> t;
    for (int i = 0; i < count; ++i)
        t.push_back({time_ms(call, rocblas_gemm_algo_solution_index, sols[i], stream, 1), sols[i]});
    std::sort(t.begin(), t.end());
    // finalists: best 4 screened + the untuned path, timed in 3 interleaved rounds of 5 reps,
    // min per candidate (clock noise on a shared box made single timings pick losers)
    std::vector<int> fin;
    for (int i = 0; i < (int) t.size() && i < 16 && fin.size() < 4; ++i)
    {
        if (t[i].first >= 1e29f) break;
        if (call.run(rocblas_gemm_algo_solution_index, t[i].second) != rocblas_status_success) continue;
        if (same()) fin.push_back(t[i].second);
    }
    std::vector<float> tm(fin.size(), 1e30f);
    float best_def = 1e30f;
    for (int r = 0; r < 3; ++r)
    {
        best_def = fminf(best_def, def_time(5));   // not std::min: HIP's int min() overload wins
        for (size_t i = 0; i < fin.size(); ++i)
            tm[i] = fminf(tm[i], time_ms(call, rocblas_gemm_algo_solution_index, fin[i], stream, 5));
    }
    float best = best_def * 0.97f;
    int best_sol = 0;
    for (size_t i = 0; i < fin.size(); ++i) if (tm[i] < best) { best = tm[i]; best_sol = fin[i]; }
    if (getenv("EXL3_DENSE_GEMM_TUNE_DEBUG"))
    {
        fprintf(stderr, "[dtune] n=%d k=%d m=%d exact=%d/%d def=%.3f", call.n, call.k, call.m,
                (int) fin.size(), count, best_def);
        for (size_t i = 0; i < fin.size(); ++i) fprintf(stderr, " %d:%.3f", fin[i], tm[i]);
        fprintf(stderr, " -> %d\n", best_sol);
    }
    return best_sol;
}

// true if handled
template <typename D>
static bool try_launch(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, cudaStream_t stream, D&& def_run)
{
    if (!enabled()) return false;
    const int k = a.size(-1);
    const int m = a.numel() / k;
    const int n = b.size(-1);
    if (m < MIN_M) return false;
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kHalf && c.dtype() != at::kFloat) return false;
    if (b.dim() != 2 || !b.is_contiguous() || c.stride(-1) != 1) return false;
    int64_t lda = 0;
    if (!a_row_pitch(a, lda)) return false;
    const int64_t ldc = c.stride(-2);
    if (ldc > std::numeric_limits<int>::max()) return false;
    int device;
    cudaGetDevice(&device);
    if (device < 0 || device >= NUM_DEV) return false;

    std::lock_guard<std::mutex> lock(g_mutex);
    load(device);
    const bool f32 = c.dtype() == at::kFloat;
    // bit 1 of the ldc field: padded A pitch (its winner is screened on the padded layout)
    Key key(n, k, m_class(m), (ldc == n ? 1 : 0) | (lda != k ? 2 : 0), f32 ? 1 : 0);
    Call call{handle(device), m, n, k, (int) ldc, (int) lda, a.data_ptr(), b.data_ptr(), c.data_ptr(),
              f32 ? rocblas_datatype_f32_r : rocblas_datatype_f16_r};
    rocblas_set_stream(call.h, stream);
    auto it = g_cache.find(key);
    int sol;
    if (it != g_cache.end()) sol = it->second;
    else
    {
        hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
        hipStreamIsCapturing(stream, &cs);
        if (cs != hipStreamCaptureStatusNone) return false;
        def_run();
        at::Tensor ref = c.clone();
        sol = tune(call, stream, [&](int reps){ return time_fn([&]{ def_run(); return true; }, stream, reps); },
                   [&]{ return c.equal(ref); });
        ref = at::Tensor();
        g_cache[key] = sol;
        store(key, sol);
    }
    if (sol == 0) return false;   // caller runs the untuned path
    // final run writes the real result (tuning reps overwrite c with the same product)
    if (call.run(rocblas_gemm_algo_solution_index, sol) != rocblas_status_success) return false;
    return true;
}

} // namespace dtune

/*
Skinny fp16 GEMM for ROCm: a [m, k] @ b [k, n] -> c [m, n] fp32/fp16, tiny m (decode rows) and
tiny n. hipblaslt answers these shapes (the GDN b/a projections, k=2560 n=48, m<=8) with a
split-K solution plus a PostGSU reduce, ~60 us for a 245 KB weight that streams in ~1 us.
36 GDN layers x 2 calls per forward made that ~6% of decode device time on gfx1151.

Split-K: grid (n/32, k/128) blocks write fp32 partials to a per-device workspace, then a
one-block reduce sums them in fixed order (deterministic, no atomics). The m rows share every
b load (b is the traffic, a is L2-resident), so cost is ~flat in m. A first version with one
block per 32 columns and no k-split was latency-bound at ~85 us (2 blocks on 20 CUs, 320
dependent load rounds per lane) - slower than hipblaslt. Parallelism first, then bandwidth.
*/

namespace skinny
{

constexpr int COLS = 32;         // output columns per block (one lane each)
constexpr int WARPS = 8;         // warps per block, each a k-sub-slice of the block's k-chunk
constexpr int THREADS = WARPS * 32;
constexpr int MAX_M = 8;
constexpr int MAX_N = 128;       // above this, let hipblaslt have it
constexpr int K_CHUNK = 128;     // k rows per block -> 16 k's per warp, 20 blocks on k=2560
constexpr int MAX_KSPLIT = 64;   // k <= 8192
constexpr int NUM_DEV = MAX_DEVICES;   // from exl3_devctx.cuh

// Partials workspace [KSPLIT][M][N] fp32 per device, allocated once. Stream-ordered use only
// (the partials kernel and the reduce run back to back on the caller's stream), like the
// cuBLAS workspace this path replaces.
static float* g_partials[NUM_DEV] = {};
static int* g_counters[NUM_DEV] = {};   // one per 32-column group, zeroed once; the last block resets its own

static int* counters(int device)
{
    if (device < 0 || device >= NUM_DEV) return nullptr;
    if (!g_counters[device])
    {
        cuda_check(cudaMalloc((void**) &g_counters[device], (MAX_N / COLS) * sizeof(int)));
        cuda_check(cudaMemset(g_counters[device], 0, (MAX_N / COLS) * sizeof(int)));
        cuda_check(cudaDeviceSynchronize());
    }
    return g_counters[device];
}

static float* partials(int device)
{
    if (device < 0 || device >= NUM_DEV) return nullptr;
    if (!g_partials[device])
    {
        cuda_check(cudaMalloc((void**) &g_partials[device], (size_t) MAX_KSPLIT * MAX_M * MAX_N * sizeof(float)));
    }
    return g_partials[device];
}

// grid (ceil(n / 32), ksplit). Block (x, y) covers columns [32x, 32x+32) and k rows
// [K_CHUNK*y, K_CHUNK*(y+1)); warp w takes every WARPS-th k of that chunk so the 8 warps
// stream interleaved 64-byte rows. Each lane holds M fp32 accumulators and issues its loads
// back to back (the k loop is fully unrolled: K_CHUNK / WARPS = 16 independent loads in
// flight per lane), which is what the one-block-per-32-columns version lacked.
//
// ONEPASS (r35): the last block to finish a column group (atomic ticket) sums that group's
// k-split partials itself in skinny_reduce_kernel's fixed y order, so the result is bit-identical
// to the two-launch form without the reduce launch. Every writer fences before the ticket; the
// last block fences (acquire) before reading. It resets its counter for the next launch.
template <int M, bool ONEPASS, typename C_T>
__global__ __launch_bounds__(THREADS)
void skinny_partials_kernel
(
    const half* __restrict__ a,   // [M, k], row stride lda
    const half* __restrict__ b,   // [k, n], row stride n
    float* __restrict__ part,     // [ksplit, M, MAX_N]
    const int k,
    const int n,
    const int lda,
    int* __restrict__ cnt,        // ONEPASS: [MAX_N / COLS] tickets
    C_T* __restrict__ c,          // ONEPASS: output [M, ldc]
    const int ldc
)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int col = blockIdx.x * COLS + lane;
    const bool active = col < n;
    const int kbase = blockIdx.y * K_CHUNK;

    float acc[M];
    #pragma unroll
    for (int i = 0; i < M; ++i) acc[i] = 0.0f;

    if (active)
    {
        constexpr int PER_WARP = K_CHUNK / WARPS;
        float bv[PER_WARP];
        #pragma unroll
        for (int j = 0; j < PER_WARP; ++j)
        {
            const int kk = kbase + j * WARPS + warp;
            bv[j] = kk < k ? __half2float(b[(size_t) kk * n + col]) : 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < PER_WARP; ++j)
        {
            const int kk = kbase + j * WARPS + warp;
            if (kk < k)
            {
                #pragma unroll
                for (int i = 0; i < M; ++i)
                    acc[i] = fmaf(__half2float(a[(size_t) i * lda + kk]), bv[j], acc[i]);
            }
        }
    }

    __shared__ float red[WARPS][M][COLS + 1];    // 33: coprime with the 32 dword banks
    #pragma unroll
    for (int i = 0; i < M; ++i) red[warp][i][lane] = acc[i];
    __syncthreads();

    if (warp == 0 && active)
    {
        float* out = part + ((size_t) blockIdx.y * M) * MAX_N;
        #pragma unroll
        for (int i = 0; i < M; ++i)
        {
            float s = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) s += red[w][i][lane];
            out[(size_t) i * MAX_N + col] = s;
        }
    }

    if constexpr (ONEPASS)
    {
        __shared__ int last_s;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            const int t = atomicAdd(&cnt[blockIdx.x], 1);
            last_s = t == (int) gridDim.y - 1;
        }
        __syncthreads();
        if (!last_s) return;
        __threadfence();
        const int ksplit = gridDim.y;
        for (int idx = threadIdx.x; idx < M * COLS; idx += THREADS)
        {
            const int i = idx / COLS;
            const int cc = blockIdx.x * COLS + idx % COLS;
            if (cc >= n) continue;
            float s = 0.0f;
            for (int y = 0; y < ksplit; ++y)
                s += __builtin_nontemporal_load(&part[((size_t) y * M + i) * MAX_N + cc]);
            if constexpr (std::is_same_v<C_T, float>)
                c[(size_t) i * ldc + cc] = s;
            else
                c[(size_t) i * ldc + cc] = __float2half(s);
        }
        if (threadIdx.x == 0) cnt[blockIdx.x] = 0;
    }
}

// One block, thread per (row, col): fixed-order sum over the k-splits -> deterministic.
template <typename C_T>
__global__ void skinny_reduce_kernel
(
    const float* __restrict__ part,   // [ksplit, M, MAX_N]
    C_T* __restrict__ c,
    const int m,
    const int n,
    const int ksplit,
    const int ldc
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    const int i = idx / n, col = idx % n;
    float s = 0.0f;
    for (int y = 0; y < ksplit; ++y)
        s += part[((size_t) y * m + i) * MAX_N + col];
    if constexpr (std::is_same_v<C_T, float>)
        c[(size_t) i * ldc + col] = s;
    else
        c[(size_t) i * ldc + col] = __float2half(s);
}

template <bool ONEPASS, typename C_T>
static void launch_partials(int m, const half* a, const half* b, float* part, int k, int n, int lda,
                            dim3 grid, cudaStream_t stream, int* cnt = nullptr, C_T* c = nullptr, int ldc = 0)
{
    #define SK_CASE(MM) case MM: skinny_partials_kernel<MM, ONEPASS, C_T><<<grid, THREADS, 0, stream>>>(a, b, part, k, n, lda, cnt, c, ldc); break;
    switch (m)
    {
        SK_CASE(1) SK_CASE(2) SK_CASE(3) SK_CASE(4) SK_CASE(5) SK_CASE(6) SK_CASE(7) SK_CASE(8)
        default: break;
    }
    #undef SK_CASE
}

// EXL3_SKINNY_1PASS (default 1): fold the reduce into the partials launch. Runtime-settable
// (skinny_set_onepass) so one process can A/B it; graphs bake the choice at capture.
static int g_onepass = -1;
static bool onepass()
{
    if (g_onepass < 0)
    {
        const char* e = getenv("EXL3_SKINNY_1PASS");
        g_onepass = (e && *e == '0') ? 0 : 1;
    }
    return g_onepass == 1;
}

static bool enabled()
{
    static int cached = -1;
    if (cached < 0)
    {
        const char* e = getenv("EXL3_HIP_SKINNY_GEMM");
        cached = (e && *e == '0') ? 0 : 1;
    }
    return cached == 1;
}

// true if handled
static bool try_launch(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, cudaStream_t stream)
{
    if (!enabled()) return false;
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kFloat && c.dtype() != at::kHalf) return false;
    if (b.dim() != 2 || !b.is_contiguous()) return false;
    const int k = a.size(-1);
    const int m = a.numel() / k;
    const int n = b.size(-1);
    if (m < 1 || m > MAX_M || n < 1 || n > MAX_N || k < 1) return false;
    const int ksplit = (k + K_CHUNK - 1) / K_CHUNK;
    if (ksplit > MAX_KSPLIT) return false;
    if (a.stride(-1) != 1 || c.stride(-1) != 1) return false;
    if (a.dim() > 2 && !a.is_contiguous()) return false;
    const int lda = a.dim() == 1 ? k : (int) a.stride(-2);
    const int ldc = (int) c.stride(-2);
    float* part = partials(a.device().index());
    if (!part) return false;
    int* cnt = counters(a.device().index());   // allocated on the first (eager) call, never under capture

    dim3 grid((n + COLS - 1) / COLS, ksplit);
    if (onepass())
    {
        if (cnt)
        {
            if (c.dtype() == at::kFloat)
                launch_partials<true, float>(m, (const half*) a.data_ptr(), (const half*) b.data_ptr(), part, k, n, lda, grid, stream,
                                             cnt, (float*) c.data_ptr(), ldc);
            else
                launch_partials<true, half>(m, (const half*) a.data_ptr(), (const half*) b.data_ptr(), part, k, n, lda, grid, stream,
                                            cnt, (half*) c.data_ptr(), ldc);
            cuda_check(cudaPeekAtLastError());
            return true;
        }
    }
    launch_partials<false, float>(m, (const half*) a.data_ptr(), (const half*) b.data_ptr(), part, k, n, lda, grid, stream);
    const int total = m * n;
    const int rthreads = total < 256 ? ((total + 31) / 32) * 32 : 256;
    const int rblocks = (total + rthreads - 1) / rthreads;
    if (c.dtype() == at::kFloat)
        skinny_reduce_kernel<float><<<rblocks, rthreads, 0, stream>>>(part, (float*) c.data_ptr(), m, n, ksplit, ldc);
    else
        skinny_reduce_kernel<half><<<rblocks, rthreads, 0, stream>>>(part, (half*) c.data_ptr(), m, n, ksplit, ldc);
    cuda_check(cudaPeekAtLastError());
    return true;
}

// skinny_cat (kda-dec): several small-N projections of the SAME a, concatenated along n into one
// weight, in ONE launch. Each column's reduction order is exactly skinny_partials_kernel's
// (16 interleaved k per warp, warps summed 0..7, k-splits summed in y order), independent of n,
// so every output is bit-identical to a separate skinny launch per projection. Each column
// range [start[s], start[s+1]) lands in its own [M, width] output, fp16 (RNE) or fp32.
constexpr int CAT_MAX_N = 512;
constexpr int CAT_MAX_SEG = 4;
static float* g_cat_partials[NUM_DEV] = {};
static int* g_cat_counters[NUM_DEV] = {};

struct CatSeg
{
    void* ptr[CAT_MAX_SEG];
    const half* w[CAT_MAX_SEG];   // segment s weight, column 0 (concat: b + start[s]; separate: its own tensor)
    int ldw[CAT_MAX_SEG];         // row stride of w[s] (concat: n; separate: segment width)
    int is_half[CAT_MAX_SEG];
    int start[CAT_MAX_SEG + 1];
    int nseg;
};

template <int M>
__global__ __launch_bounds__(THREADS)
void skinny_cat_kernel
(
    const half* __restrict__ a,
    float* __restrict__ part,     // [ksplit, M, CAT_MAX_N]
    const int k,
    const int n,
    const int lda,
    int* __restrict__ cnt,
    const CatSeg seg
)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int col = blockIdx.x * COLS + lane;
    const bool active = col < n;
    const int kbase = blockIdx.y * K_CHUNK;

    float acc[M];
    #pragma unroll
    for (int i = 0; i < M; ++i) acc[i] = 0.0f;

    if (active)
    {
        constexpr int PER_WARP = K_CHUNK / WARPS;
        int cs = 0;
        while (cs + 1 < seg.nseg && col >= seg.start[cs + 1]) ++cs;
        const half* bw = seg.w[cs] + (col - seg.start[cs]);
        const int ldb = seg.ldw[cs];
        float bv[PER_WARP];
        #pragma unroll
        for (int j = 0; j < PER_WARP; ++j)
        {
            const int kk = kbase + j * WARPS + warp;
            bv[j] = kk < k ? __half2float(bw[(size_t) kk * ldb]) : 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < PER_WARP; ++j)
        {
            const int kk = kbase + j * WARPS + warp;
            if (kk < k)
            {
                #pragma unroll
                for (int i = 0; i < M; ++i)
                    acc[i] = fmaf(__half2float(a[(size_t) i * lda + kk]), bv[j], acc[i]);
            }
        }
    }

    __shared__ float red[WARPS][M][COLS + 1];
    #pragma unroll
    for (int i = 0; i < M; ++i) red[warp][i][lane] = acc[i];
    __syncthreads();

    if (warp == 0 && active)
    {
        float* out = part + ((size_t) blockIdx.y * M) * CAT_MAX_N;
        #pragma unroll
        for (int i = 0; i < M; ++i)
        {
            float s = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) s += red[w][i][lane];
            out[(size_t) i * CAT_MAX_N + col] = s;
        }
    }

    __shared__ int last_s;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int t = atomicAdd(&cnt[blockIdx.x], 1);
        last_s = t == (int) gridDim.y - 1;
    }
    __syncthreads();
    if (!last_s) return;
    __threadfence();
    const int ksplit = gridDim.y;
    for (int idx = threadIdx.x; idx < M * COLS; idx += THREADS)
    {
        const int i = idx / COLS;
        const int cc = blockIdx.x * COLS + idx % COLS;
        if (cc >= n) continue;
        float s = 0.0f;
        for (int y = 0; y < ksplit; ++y)
            s += __builtin_nontemporal_load(&part[((size_t) y * M + i) * CAT_MAX_N + cc]);
        int sg = 0;
        while (sg + 1 < seg.nseg && cc >= seg.start[sg + 1]) ++sg;
        const int w = seg.start[sg + 1] - seg.start[sg];
        const int lc = cc - seg.start[sg];
        if (seg.is_half[sg])
            ((half*) seg.ptr[sg])[(size_t) i * w + lc] = __float2half(s);
        else
            ((float*) seg.ptr[sg])[(size_t) i * w + lc] = s;
    }
    if (threadIdx.x == 0) cnt[blockIdx.x] = 0;
}

} // namespace skinny

// a: [M, k] fp16 (M <= 8, contiguous rows). Weights either one concatenated b: [k, n] (skinny_cat) or
// one contiguous [k, w_s] tensor per output (skinny_cat_w: no concat copy, same values in the same order,
// so bit-identical). n = sum(w_s) <= 512, k <= 8192. outs: contiguous [M, w_s] fp16 or fp32, column order.
static void skinny_cat_impl(at::Tensor a, const at::Tensor* b, const std::vector<at::Tensor>* ws,
                            std::vector<at::Tensor> outs)
{
    using namespace skinny;
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK(a.dim() >= 1 && a.size(-1) >= 1, "skinny_cat: a must have non-zero K");
    const int k = a.size(-1);
    const int m = a.numel() / k;
    TORCH_CHECK(a.is_contiguous(), "skinny_cat: a must be contiguous");
    TORCH_CHECK((int) outs.size() >= 1 && (int) outs.size() <= CAT_MAX_SEG, "skinny_cat: 1..4 outputs");
    TORCH_CHECK(!ws || ws->size() == outs.size(), "skinny_cat_w: one weight per output");
    if (m == 0) return;
    CatSeg seg;
    seg.nseg = (int) outs.size();
    int acc = 0;
    for (int s = 0; s < seg.nseg; ++s)
    {
        const at::Tensor& o = outs[s];
        TORCH_CHECK(o.is_contiguous(), "skinny_cat: outputs must be contiguous");
        TORCH_CHECK(o.dtype() == at::kHalf || o.dtype() == at::kFloat, "skinny_cat: outputs fp16/fp32");
        const int w = (int) (o.numel() / m);
        TORCH_CHECK((int64_t) w * m == o.numel() && o.size(-1) == w, "skinny_cat: output shape");
        seg.ptr[s] = o.data_ptr();
        seg.is_half[s] = o.dtype() == at::kHalf;
        seg.start[s] = acc;
        if (ws)
        {
            const at::Tensor& ww = (*ws)[s];
            TORCH_CHECK_DTYPE(ww, kHalf);
            TORCH_CHECK(ww.dim() == 2 && ww.is_contiguous() && ww.size(0) == k && ww.size(1) == w,
                        "skinny_cat_w: weight s must be contiguous [k, w_s]");
            seg.w[s] = (const half*) ww.data_ptr();
            seg.ldw[s] = w;
        }
        acc += w;
    }
    seg.start[seg.nseg] = acc;
    const int n = acc;
    if (b)
    {
        TORCH_CHECK_DTYPE((*b), kHalf);
        TORCH_CHECK(b->dim() == 2 && b->is_contiguous(), "skinny_cat: b must be contiguous [k, n]");
        TORCH_CHECK(b->size(0) == k, "skinny_cat: k mismatch");
        TORCH_CHECK(b->size(1) == n, "skinny_cat: output widths must sum to n");
        for (int s = 0; s < seg.nseg; ++s)
        {
            seg.w[s] = (const half*) b->data_ptr() + seg.start[s];
            seg.ldw[s] = n;
        }
    }
    TORCH_CHECK(m >= 1 && m <= MAX_M && n >= 1 && n <= CAT_MAX_N, "skinny_cat: shape out of range");
    const int ksplit = (k + K_CHUNK - 1) / K_CHUNK;
    TORCH_CHECK(ksplit <= MAX_KSPLIT, "skinny_cat: k too large");

    const int dev = a.device().index();
    TORCH_CHECK(dev >= 0 && dev < NUM_DEV, "skinny_cat: device");
    if (!g_cat_partials[dev])
        cuda_check(cudaMalloc((void**) &g_cat_partials[dev], (size_t) MAX_KSPLIT * MAX_M * CAT_MAX_N * sizeof(float)));
    if (!g_cat_counters[dev])
    {
        cuda_check(cudaMalloc((void**) &g_cat_counters[dev], (CAT_MAX_N / COLS) * sizeof(int)));
        cuda_check(cudaMemset(g_cat_counters[dev], 0, (CAT_MAX_N / COLS) * sizeof(int)));
        cuda_check(cudaDeviceSynchronize());
    }
    dim3 grid((n + COLS - 1) / COLS, ksplit);
    const half* ap = (const half*) a.data_ptr();
    #define SC_CASE(MM) case MM: skinny_cat_kernel<MM><<<grid, THREADS, 0, stream>>>(ap, g_cat_partials[dev], k, n, k, g_cat_counters[dev], seg); break;
    switch (m)
    {
        SC_CASE(1) SC_CASE(2) SC_CASE(3) SC_CASE(4) SC_CASE(5) SC_CASE(6) SC_CASE(7) SC_CASE(8)
        default: break;
    }
    #undef SC_CASE
    cuda_check(cudaPeekAtLastError());
}

void skinny_cat(at::Tensor a, at::Tensor b, std::vector<at::Tensor> outs) { skinny_cat_impl(a, &b, nullptr, outs); }
void skinny_cat_w(at::Tensor a, std::vector<at::Tensor> ws, std::vector<at::Tensor> outs) { skinny_cat_impl(a, nullptr, &ws, outs); }

void skinny_set_onepass(int64_t on) { skinny::g_onepass = on ? 1 : 0; }

/*
wide: M <= 8 rows against a wide row-major fp16 weight [k, n] (n > skinny::MAX_N, n % 8 == 0). At decode the
weight read is the whole cost; hipBLASLt's tile GEMMs reach ~135 GB/s on these shapes (GLM-5.3-Flash BF16
attention / shared expert / head: ~113 ms of a 144 ms token). Each lane streams 16-byte rows of 8 columns, a
block covers 256 columns x a k-chunk with its 8 warps on interleaved rows, the a rows of the chunk sit in LDS,
and k-split partials are summed in fixed y order by a second launch: deterministic, independent of timing.
EXL3_HIP_WIDE_GEMV=0 disables.
*/
namespace wide
{

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int VEC = 8;                       // columns per lane (one 16-byte load); 4 (8-byte loads) for M >= 5
constexpr int COLS = 32 * VEC;               // columns per block at VEC = 8
constexpr int MAX_M = 8;
constexpr int MAX_KC = 1024;                 // k rows per block (LDS: MAX_M * MAX_KC halves = 16 KiB)
constexpr int TARGET_BLOCKS = 160;           // ~4 blocks per CU on 40 CUs (sweep: 160 > 320 > 640 for M = 1 and 8)
constexpr size_t WS_BYTES = 48u << 20;       // partials workspace per device
static float* g_ws[MAX_DEVICES] = {};

static bool enabled()
{
    static int cached = -1;
    if (cached < 0)
    {
        const char* e = getenv("EXL3_HIP_WIDE_GEMV");
        cached = (e && *e == '0') ? 0 : 1;
    }
    return cached == 1;
}

template <int M, int V>
__global__ __launch_bounds__(THREADS)
void wide_partials_kernel
(
    const half* __restrict__ a,   // [M, k], row stride lda
    const half* __restrict__ b,   // [k, n]
    float* __restrict__ part,     // [ksplit, M, n]
    const int k,
    const int n,
    const int lda,
    const int kc                  // k rows per block, multiple of WARPS
)
{
    __shared__ half sa[M][MAX_KC];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int k0 = blockIdx.y * kc;
    const int k1 = min(k0 + kc, k);
    for (int idx = threadIdx.x; idx < M * kc; idx += THREADS)
    {
        const int i = idx / kc, kk = idx % kc;
        sa[i][kk] = (k0 + kk < k) ? a[(size_t) i * lda + k0 + kk] : __float2half(0.0f);
    }
    __syncthreads();

    constexpr int BC = 32 * V;                // columns per block
    using LT = std::conditional_t<V == 8, uint4, uint2>;
    const int col = blockIdx.x * BC + lane * V;
    const bool active = col < n;
    float acc[M][V];
    #pragma unroll
    for (int i = 0; i < M; ++i)
        #pragma unroll
        for (int v = 0; v < V; ++v) acc[i][v] = 0.0f;

    if (active)
    {
        const half* bp = b + col;
        int kk = k0 + warp;
        // four rows in flight per lane
        constexpr int Q = V / 2;              // half2 per load
        constexpr int U = V == 8 ? 4 : 8;     // rows in flight per lane
        for (; kk + (U - 1) * WARPS < k1; kk += U * WARPS)
        {
            LT r[U];
            #pragma unroll
            for (int u = 0; u < U; ++u) r[u] = *((const LT*) (bp + (size_t) (kk + u * WARPS) * n));
            #pragma unroll
            for (int u = 0; u < U; ++u)
            {
                const half2* h = (const half2*) &r[u];
                float2 f[Q];
                #pragma unroll
                for (int q = 0; q < Q; ++q) f[q] = __half22float2(h[q]);
                #pragma unroll
                for (int i = 0; i < M; ++i)
                {
                    const float av = __half2float(sa[i][kk + u * WARPS - k0]);
                    #pragma unroll
                    for (int q = 0; q < Q; ++q)
                    {
                        acc[i][2 * q] = fmaf(av, f[q].x, acc[i][2 * q]);
                        acc[i][2 * q + 1] = fmaf(av, f[q].y, acc[i][2 * q + 1]);
                    }
                }
            }
        }
        for (; kk < k1; kk += WARPS)
        {
            const LT r = *((const LT*) (bp + (size_t) kk * n));
            const half2* h = (const half2*) &r;
            #pragma unroll
            for (int i = 0; i < M; ++i)
            {
                const float av = __half2float(sa[i][kk - k0]);
                #pragma unroll
                for (int q = 0; q < Q; ++q)
                {
                    const float2 f = __half22float2(h[q]);
                    acc[i][2 * q] = fmaf(av, f.x, acc[i][2 * q]);
                    acc[i][2 * q + 1] = fmaf(av, f.y, acc[i][2 * q + 1]);
                }
            }
        }
    }

    // reduce the 8 warps' row slices through LDS, one row of M at a time (fixed order)
    __shared__ float red[WARPS][BC];
    float* out = part + (size_t) blockIdx.y * M * n;
    #pragma unroll
    for (int i = 0; i < M; ++i)
    {
        #pragma unroll
        for (int v = 0; v < V; ++v) red[warp][lane * V + v] = acc[i][v];
        __syncthreads();
        for (int c = threadIdx.x; c < BC; c += THREADS)
        {
            const int gc = blockIdx.x * BC + c;
            if (gc < n)
            {
                float s = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) s += red[w][c];
                out[(size_t) i * n + gc] = s;
            }
        }
        __syncthreads();
    }
}

template <typename C_T>
__global__ void wide_reduce_kernel(const float* __restrict__ part, C_T* __restrict__ c, int m, int n, int ksplit, int ldc)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    const int i = idx / n, col = idx % n;
    float s = 0.0f;
    for (int y = 0; y < ksplit; ++y) s += part[((size_t) y * m + i) * n + col];
    if constexpr (std::is_same_v<C_T, float>) c[(size_t) i * ldc + col] = s;
    else c[(size_t) i * ldc + col] = __float2half(s);
}

static bool try_launch(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, cudaStream_t stream)
{
    if (!enabled()) return false;
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kFloat && c.dtype() != at::kHalf) return false;
    if (b.dim() != 2 || !b.is_contiguous()) return false;
    const int k = a.size(-1);
    const int m = a.numel() / k;
    const int n = b.size(-1);
    if (m < 1 || m > MAX_M || n <= skinny::MAX_N || n % VEC || k < 64) return false;
    if (((uintptr_t) b.data_ptr()) % 16) return false;
    if (a.stride(-1) != 1 || c.stride(-1) != 1) return false;
    if (a.dim() > 2 && !a.is_contiguous()) return false;
    const int lda = a.dim() == 1 ? k : (int) a.stride(-2);
    const int ldc = c.dim() == 1 ? n : (int) c.stride(-2);
    const int dev = a.device().index();
    if (dev < 0 || dev >= MAX_DEVICES) return false;
    if (!g_ws[dev])
    {
        cudaStreamCaptureStatus st;
        cuda_check(cudaStreamIsCapturing(stream, &st));
        if (st != cudaStreamCaptureStatusNone) return false;
        cuda_check(cudaMalloc((void**) &g_ws[dev], WS_BYTES));
    }
    // M >= 5: 4 columns per lane halves the accumulators (more loads in flight); EXL3_WIDE_V4_MIN_M overrides
    static int v4_min = -1;
    if (v4_min < 0) { const char* e = getenv("EXL3_WIDE_V4_MIN_M"); v4_min = e ? atoi(e) : 99; }   // off: measured slower on gfx1151
    const bool v4 = m >= v4_min;
    const int bcols = v4 ? 32 * 4 : COLS;
    const int cblocks = (n + bcols - 1) / bcols;
    static int tblocks = -1;
    if (tblocks < 0) { const char* e = getenv("EXL3_WIDE_BLOCKS"); tblocks = e ? std::max(1, atoi(e)) : TARGET_BLOCKS; }
    int ksplit = std::max(1, tblocks / cblocks);
    int kc = (k + ksplit - 1) / ksplit;
    kc = std::min(MAX_KC, std::max(64, ((kc + 4 * WARPS - 1) / (4 * WARPS)) * (4 * WARPS)));
    ksplit = (k + kc - 1) / kc;
    if ((size_t) ksplit * m * n * sizeof(float) > WS_BYTES) return false;
    float* part = g_ws[dev];
    dim3 grid(cblocks, ksplit);
    #define WD_CASE(MM) case MM: \
        if (v4) wide_partials_kernel<MM, 4><<<grid, THREADS, 0, stream>>>((const half*) a.data_ptr(), (const half*) b.data_ptr(), part, k, n, lda, kc); \
        else    wide_partials_kernel<MM, 8><<<grid, THREADS, 0, stream>>>((const half*) a.data_ptr(), (const half*) b.data_ptr(), part, k, n, lda, kc); \
        break;
    switch (m) { WD_CASE(1) WD_CASE(2) WD_CASE(3) WD_CASE(4) WD_CASE(5) WD_CASE(6) WD_CASE(7) WD_CASE(8) default: return false; }
    #undef WD_CASE
    const int total = m * n;
    const int rblocks = (total + 255) / 256;
    if (c.dtype() == at::kFloat)
        wide_reduce_kernel<float><<<rblocks, 256, 0, stream>>>(part, (float*) c.data_ptr(), m, n, ksplit, ldc);
    else
        wide_reduce_kernel<half><<<rblocks, 256, 0, stream>>>(part, (half*) c.data_ptr(), m, n, ksplit, ldc);
    cuda_check(cudaPeekAtLastError());
    return true;
}

/*
Q8 weights (KYOJIN_Q8_LINEAR): the same GEMV over an 8-bit weight, half the bytes of fp16. qw [k, n] uint8 holds
q + 128 (q in -127..127), qs [k / Q8_G, n] fp16 the scale of each 64-row group of a column; w = (qw - 128) * s.
Each iteration of a warp covers the 64 rows of one group (8 warps x 8 rows in flight), so a lane loads its 8 column
scales once per group. Same fixed-order k-split as the fp16 kernel: deterministic.
*/
constexpr int Q8_G = 64;

template <int M, bool RNG = false>
__global__ __launch_bounds__(THREADS)
void wide_q8_partials_kernel
(
    const half* __restrict__ a,       // [M, k], row stride lda
    const uint8_t* __restrict__ qw,   // [k, n]
    const half* __restrict__ qs,      // [k / Q8_G, n]
    float* __restrict__ part,         // [ksplit, M, n]
    const int k,
    const int n,
    const int lda,
    const int kc,                     // k rows per block, multiple of Q8_G
    const int* __restrict__ rng = nullptr,  // RNG: device [lo, hi), multiples of Q8_G (and of V columns)
    const int rmode = 0               // RNG: 0 = only output columns [lo, hi) (the others come out 0), 1 = only k rows [lo, hi)
)
{
    __shared__ half sa[M][MAX_KC];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int k0 = blockIdx.y * kc;
    const int k1 = min(k0 + kc, k);
    for (int idx = threadIdx.x; idx < M * kc; idx += THREADS)
    {
        const int i = idx / kc, kk = idx % kc;
        sa[i][kk] = (k0 + kk < k) ? a[(size_t) i * lda + k0 + kk] : __float2half(0.0f);
    }
    __syncthreads();

    constexpr int V = 8;
    constexpr int BC = 32 * V;
    constexpr int U = Q8_G / WARPS;           // rows of a group per warp
    const int col = blockIdx.x * BC + lane * V;
    bool active = col < n;
    int g_lo = k0, g_hi = k1;
    if constexpr (RNG)
    {
        const int lo = rng[0], hi = rng[1];
        if (rmode == 0) active = active && col >= lo && col < hi;
        else { g_lo = max(k0, lo); g_hi = min(k1, hi); }
    }
    float acc[M][V];
    #pragma unroll
    for (int i = 0; i < M; ++i)
        #pragma unroll
        for (int v = 0; v < V; ++v) acc[i][v] = 0.0f;

    if (active)
    {
        const uint8_t* bp = qw + col;
        for (int g0 = g_lo; g0 < g_hi; g0 += Q8_G)
        {
            const uint4 sr = *((const uint4*) (qs + (size_t) (g0 / Q8_G) * n + col));
            const half2* sh = (const half2*) &sr;
            float s[V], o[V];
            #pragma unroll
            for (int q = 0; q < V / 2; ++q)
            {
                const float2 f = __half22float2(sh[q]);
                s[2 * q] = f.x; s[2 * q + 1] = f.y;
            }
            #pragma unroll
            for (int v = 0; v < V; ++v) o[v] = -128.0f * s[v];
            uint2 r[U];
            #pragma unroll
            for (int u = 0; u < U; ++u) r[u] = *((const uint2*) (bp + (size_t) (g0 + warp + u * WARPS) * n));
            #pragma unroll
            for (int u = 0; u < U; ++u)
            {
                float w[V];
                #pragma unroll
                for (int v = 0; v < 4; ++v)
                {
                    w[v] = fmaf((float) ((r[u].x >> (8 * v)) & 0xff), s[v], o[v]);
                    w[v + 4] = fmaf((float) ((r[u].y >> (8 * v)) & 0xff), s[v + 4], o[v + 4]);
                }
                const int kr = g0 + warp + u * WARPS - k0;
                #pragma unroll
                for (int i = 0; i < M; ++i)
                {
                    const float av = __half2float(sa[i][kr]);
                    #pragma unroll
                    for (int v = 0; v < V; ++v) acc[i][v] = fmaf(av, w[v], acc[i][v]);
                }
            }
        }
    }

    __shared__ float red[WARPS][BC];
    float* out = part + (size_t) blockIdx.y * M * n;
    #pragma unroll
    for (int i = 0; i < M; ++i)
    {
        #pragma unroll
        for (int v = 0; v < V; ++v) red[warp][lane * V + v] = acc[i][v];
        __syncthreads();
        for (int c = threadIdx.x; c < BC; c += THREADS)
        {
            const int gc = blockIdx.x * BC + c;
            if (gc < n)
            {
                float s = 0.0f;
                #pragma unroll
                for (int w = 0; w < WARPS; ++w) s += red[w][c];
                out[(size_t) i * n + gc] = s;
            }
        }
        __syncthreads();
    }
}

__global__ void q8_dequant_kernel(const uint8_t* __restrict__ qw, const half* __restrict__ qs, half* __restrict__ out, int k, int n)
{
    const size_t idx = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx >= (size_t) k * n) return;
    const int row = idx / n, col = idx % n;
    const uint2 r = *((const uint2*) (qw + idx));
    const uint4 sr = *((const uint4*) (qs + (size_t) (row / Q8_G) * n + col));
    const half* sh = (const half*) &sr;
    half o[8];
    #pragma unroll
    for (int v = 0; v < 4; ++v)
    {
        const float s0 = __half2float(sh[v]), s1 = __half2float(sh[v + 4]);
        o[v] = __float2half(fmaf((float) ((r.x >> (8 * v)) & 0xff), s0, -128.0f * s0));
        o[v + 4] = __float2half(fmaf((float) ((r.y >> (8 * v)) & 0xff), s1, -128.0f * s1));
    }
    *((uint4*) (out + idx)) = *((uint4*) o);
}

static bool try_launch_q8(const at::Tensor& a, const at::Tensor& qw, const at::Tensor& qs, at::Tensor& c, cudaStream_t stream,
                          const int* rng = nullptr, int rmode = 0)
{
    if (a.dtype() != at::kHalf || qw.dtype() != at::kByte || qs.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kFloat && c.dtype() != at::kHalf) return false;
    const int k = a.size(-1);
    const int m = a.numel() / k;
    const int n = qw.size(-1);
    if (m < 1 || m > MAX_M || n % VEC || k % Q8_G || qw.size(0) != k) return false;
    if (((uintptr_t) qw.data_ptr()) % 16 || ((uintptr_t) qs.data_ptr()) % 16) return false;
    if (a.stride(-1) != 1 || c.stride(-1) != 1) return false;
    if (a.dim() > 2 && !a.is_contiguous()) return false;
    const int lda = a.dim() == 1 ? k : (int) a.stride(-2);
    const int ldc = c.dim() == 1 ? n : (int) c.stride(-2);
    const int dev = a.device().index();
    if (dev < 0 || dev >= MAX_DEVICES) return false;
    if (!g_ws[dev])
    {
        cudaStreamCaptureStatus st;
        cuda_check(cudaStreamIsCapturing(stream, &st));
        if (st != cudaStreamCaptureStatusNone) return false;
        cuda_check(cudaMalloc((void**) &g_ws[dev], WS_BYTES));
    }
    const int cblocks = (n + COLS - 1) / COLS;
    static int tblocks = -1;
    if (tblocks < 0) { const char* e = getenv("EXL3_Q8_BLOCKS"); tblocks = e ? std::max(1, atoi(e)) : TARGET_BLOCKS; }
    int ksplit = std::max(1, tblocks / cblocks);
    int kc = (k + ksplit - 1) / ksplit;
    kc = std::min(MAX_KC, std::max(Q8_G, ((kc + Q8_G - 1) / Q8_G) * Q8_G));
    ksplit = (k + kc - 1) / kc;
    if ((size_t) ksplit * m * n * sizeof(float) > WS_BYTES) return false;
    float* part = g_ws[dev];
    dim3 grid(cblocks, ksplit);
    #define WQ_CASE(MM) case MM: \
        if (rng) { \
            wide_q8_partials_kernel<MM, true><<<grid, THREADS, 0, stream>>>( \
            (const half*) a.data_ptr(), (const uint8_t*) qw.data_ptr(), (const half*) qs.data_ptr(), part, k, n, lda, kc, rng, rmode); \
        } else { \
            wide_q8_partials_kernel<MM><<<grid, THREADS, 0, stream>>>( \
            (const half*) a.data_ptr(), (const uint8_t*) qw.data_ptr(), (const half*) qs.data_ptr(), part, k, n, lda, kc); \
        } \
        break;
    switch (m) { WQ_CASE(1) WQ_CASE(2) WQ_CASE(3) WQ_CASE(4) WQ_CASE(5) WQ_CASE(6) WQ_CASE(7) WQ_CASE(8) default: return false; }
    #undef WQ_CASE
    const int rblocks = (m * n + 255) / 256;
    if (c.dtype() == at::kFloat)
        wide_reduce_kernel<float><<<rblocks, 256, 0, stream>>>(part, (float*) c.data_ptr(), m, n, ksplit, ldc);
    else
        wide_reduce_kernel<half><<<rblocks, 256, 0, stream>>>(part, (half*) c.data_ptr(), m, n, ksplit, ldc);
    cuda_check(cudaPeekAtLastError());
    return true;
}

} // namespace wide

void q8_dequant(at::Tensor qw, at::Tensor qs, at::Tensor out)
{
    const at::cuda::OptionalCUDAGuard device_guard(qw.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(qw.dtype() == at::kByte && qs.dtype() == at::kHalf && out.dtype() == at::kHalf, "q8_dequant: dtypes");
    TORCH_CHECK(qw.is_contiguous() && qs.is_contiguous() && out.is_contiguous(), "q8_dequant: contiguous");
    const int k = qw.size(0), n = qw.size(1);
    TORCH_CHECK(n % 8 == 0 && k % wide::Q8_G == 0, "q8_dequant: shape");
    const size_t threads = (size_t) k * n / 8;
    wide::q8_dequant_kernel<<<(threads + 255) / 256, 256, 0, stream>>>((const uint8_t*) qw.data_ptr(), (const half*) qs.data_ptr(), (half*) out.data_ptr(), k, n);
    cuda_check(cudaPeekAtLastError());
}

// y = a @ dequant(qw, qs): the Q8 GEMV at <= 8 rows, else dequantize to a transient fp16 weight and run hgemm
void hgemm_q8_gr(at::Tensor a, at::Tensor qw, at::Tensor qs, at::Tensor c, Graph* graph)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
    if (wide::try_launch_q8(a, qw, qs, c, stream)) return;
    TORCH_CHECK(!graph, "hgemm_q8: graph capture needs the <= 8 row Q8 GEMV path");
    at::Tensor w = at::empty({qw.size(0), qw.size(1)}, a.options().dtype(at::kHalf));
    q8_dequant(qw, qs, w);
    hgemm_gr(a, w, c, nullptr);
}

void hgemm_q8(at::Tensor a, at::Tensor qw, at::Tensor qs, at::Tensor c) { hgemm_q8_gr(a, qw, qs, c, nullptr); }

/*
KYOJIN_EP_SHARED_DYN: per-MoE-call split of the (replicated) shared expert between the two EP ranks, from the routing
both ranks already hold. sel [nsel] are router picks in this rank's slot order (own experts in [0, n_local)). Rank
loads in expert-read units: l0 / l1 = distinct experts of rank 0 / 1, rank 1 costs w1 per unit plus a fixed offset
b1 (units), one 128-channel shared block costs sb units. Rank 0 takes x blocks (rounded optimum of
max(l0 + x sb, w1 (l1 + (nb - x) sb) + b1)), rank 1 the rest; this rank's channel range goes to rng = [lo, hi).
Both ranks count the same integers, so they agree on x exactly. One block of 256 threads, n_experts <= 1024.
*/
__global__ void ep_shared_range_kernel(const int64_t* __restrict__ sel, int nsel, int n_local, int n_experts, int rank,
                                       int nb, float sb, float w1, float b1, int* __restrict__ rng)
{
    __shared__ unsigned char seen[1024];
    __shared__ int cnt[2];
    for (int i = threadIdx.x; i < n_experts; i += blockDim.x) seen[i] = 0;
    if (threadIdx.x < 2) cnt[threadIdx.x] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < nsel; i += blockDim.x)
    {
        const int64_t e = sel[i];
        if (e >= 0 && e < n_experts) seen[e] = 1;
    }
    __syncthreads();
    int mine = 0, other = 0;
    for (int i = threadIdx.x; i < n_experts; i += blockDim.x)
        if (seen[i]) { if (i < n_local) ++mine; else ++other; }
    atomicAdd(&cnt[0], mine);
    atomicAdd(&cnt[1], other);
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const float l0 = (float) (rank == 0 ? cnt[0] : cnt[1]);
        const float l1 = (float) (rank == 0 ? cnt[1] : cnt[0]);
        const float xs = (w1 * (l1 + nb * sb) + b1 - l0) / (sb * (1.0f + w1));
        const int x = min(nb, max(0, (int) floorf(xs + 0.5f)));
        rng[0] = rank == 0 ? 0 : x * 128;
        rng[1] = rank == 0 ? x * 128 : nb * 128;
    }
}

void ep_shared_range(at::Tensor sel, int64_t n_local, int64_t n_experts, int64_t rank, int64_t nb, double sb, double w1,
                     double b1, at::Tensor rng)
{
    const at::cuda::OptionalCUDAGuard device_guard(sel.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(sel.dtype() == at::kLong && sel.is_contiguous() && n_experts <= 1024, "ep_shared_range: sel");
    TORCH_CHECK(rng.dtype() == at::kInt && rng.numel() >= 2, "ep_shared_range: rng");
    ep_shared_range_kernel<<<1, 256, 0, stream>>>((const int64_t*) sel.data_ptr(), (int) sel.numel(), (int) n_local,
        (int) n_experts, (int) rank, (int) nb, (float) sb, (float) w1, (float) b1, (int*) rng.data_ptr());
    cuda_check(cudaPeekAtLastError());
}

// KYOJIN_EP_SHARED_DYN: the <= 8 row Q8 GEMV restricted to a device-side range rng = int32 [lo, hi) (no host sync):
// mode 0 computes only the output columns [lo, hi) (the others come out 0), mode 1 only the k rows [lo, hi)
void hgemm_q8_range(at::Tensor a, at::Tensor qw, at::Tensor qs, at::Tensor c, at::Tensor rng, int64_t mode)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(rng.dtype() == at::kInt && rng.is_cuda() && rng.numel() >= 2, "hgemm_q8_range: rng must be int32 cuda [2]");
    TORCH_CHECK(wide::try_launch_q8(a, qw, qs, c, stream, (const int*) rng.data_ptr(), (int) mode),
                "hgemm_q8_range: shape not supported by the Q8 GEMV path");
}

#endif // USE_ROCM

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
)
{
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();
#if defined(USE_ROCM)
    if (skinny::try_launch(a, b, c, stream)) return;
    if (wide::try_launch(a, b, c, stream)) return;
    if (!graph && dtune::try_launch(a, b, c, stream, [&]{ hgemm_gemmex_impl(a, b, c, stream); })) return;
#endif
    hgemm_gemmex_impl(a, b, c, stream);

    if (graph) graph->need_cublas = true;
}

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    hgemm_gr(a, b, c, nullptr);
}

/*
Strided-batched row-major matmul, a[b] @ w[b] -> c[b] for b in [0, B), fp16 inputs with fp32
accumulation (same cuBLAS setup as hgemm). a: [B, m, k], w: [B, k, n], c: [B, m, n], all
contiguous; c fp16 or fp32. Used by the batched expert reconstruct path (moe_batch_recon.py).
*/
void hgemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
)
{
    // Reconstruct-path GEMM: the fp16-accumulator kernel where it pays (GeForce), else cuBLAS
    if (hgemm_f16acc_try(a, w, c)) return;

    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(a, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    bool output_fp32 = c.dtype() == at::kFloat;
    TORCH_CHECK(output_fp32 || c.dtype() == at::kHalf, "hgemm_batched: c must be float32 or float16");
    TORCH_CHECK_DIM(a, 3);
    TORCH_CHECK_DIM(w, 3);
    TORCH_CHECK_DIM(c, 3);
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && c.is_contiguous(), "hgemm_batched: tensors must be contiguous");
    TORCH_CHECK_SHAPES(a, 0, w, 0, 1);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK_SHAPES(a, 2, w, 1, 1);
    TORCH_CHECK_SHAPES(a, 1, c, 1, 1);
    TORCH_CHECK_SHAPES(w, 2, c, 2, 1);

    int batch = a.size(0);
    int size_m = a.size(1);
    int size_k = a.size(2);
    int size_n = w.size(2);
    if (!batch || !size_m || !size_n || !size_k) return;

    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, stream);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    auto r = cublasGemmStridedBatchedEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, w.data_ptr(), CUDA_R_16F, size_n, (long long) size_k * size_n,
                 a.data_ptr(), CUDA_R_16F, size_k, (long long) size_m * size_k,
        &beta_,  c.data_ptr(), output_fp32 ? CUDA_R_32F : CUDA_R_16F, size_n, (long long) size_m * size_n,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}
