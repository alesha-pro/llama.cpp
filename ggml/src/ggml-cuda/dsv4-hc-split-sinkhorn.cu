#include "dsv4-hc-split-sinkhorn.cuh"

// Maximum n_hc supported (matches CPU reference assert at ops.cpp:11014 and
// the dst comb matrix scratch buffer size below).
#define DSV4_HC_SINKHORN_MAX_N_HC 16

// One block per row. Inside the block:
//   - threads cooperate (parallel for) on the pre/post slices and the final
//     copy of the comb matrix back to dst.
//   - the n_hc x n_hc Sinkhorn iterations run REGISTER-RESIDENT on the first
//     n_hc lanes (n_hc <= 16): each lane holds a full matrix row in registers,
//     row-normalize is barrier-free, column-normalize uses butterfly shuffles.
//     (Summation order inside a column reduction differs from the old
//     per-lane serial order by one pairing level; difference is ~1 ulp and
//     irrelevant for mixing weights.)
//
// The comb matrix lives in shared memory (sized for the worst case 16x16
// = 256 floats = 1 KiB per block, well within any device's shared-mem
// budget).
// ---------------- Section 3 fast path (compile-time n_hc) ----------------
// One warp per row; lane `dst_hc` owns matrix row dst_hc in REGISTERS (the
// loops are fully unrolled at compile time — dynamic indexing would spill
// c[] to local memory and cost ~1.3 us per sinkhorn iteration). Column
// reductions use butterfly shuffles (pairing order differs from the serial
// reference by ~1 ulp in the mixing weights).
template <int N_HC>
static __global__ void dsv4_hc_split_sinkhorn_fast(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float       * __restrict__ dst,
        const int   sinkhorn_iters,
        const int   n_rows,
        const int   nb01, const int nb1, const float eps) {
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }
    const int tid = threadIdx.x;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    const float * row_in  = (const float *) ((const char *) mixes + row * nb01);
    float       * row_out = (float *)       ((char *)       dst   + row * nb1);

    if (tid < N_HC) {
        const float z = row_in[tid] * pre_scale + base[tid];
        row_out[tid] = 1.0f / (1.0f + expf(-z)) + eps;
        const float z2 = row_in[N_HC + tid] * post_scale + base[N_HC + tid];
        row_out[N_HC + tid] = 2.0f / (1.0f + expf(-z2));
    }

    const unsigned full = 0xffffffffu;
    float c[N_HC];
    if (tid < N_HC) {
        #pragma unroll
        for (int s = 0; s < N_HC; ++s) {
            const int off = 2 * N_HC + tid * N_HC + s;
            c[s] = row_in[off] * comb_scale + base[off];
        }
        // first pass: softmax over the row (max-subtract, exp, normalize) + eps
        float row_max = c[0];
        #pragma unroll
        for (int s = 1; s < N_HC; ++s) row_max = fmaxf(row_max, c[s]);
        float row_sum = 0.0f;
        #pragma unroll
        for (int s = 0; s < N_HC; ++s) { c[s] = expf(c[s] - row_max); row_sum += c[s]; }
        const float inv_sum = 1.0f / row_sum;
        #pragma unroll
        for (int s = 0; s < N_HC; ++s) c[s] = c[s] * inv_sum + eps;
    }
    // first column-normalize (butterfly over lanes; all threads run the shuffles)
    #pragma unroll
    for (int s = 0; s < N_HC; ++s) {
        float sum = tid < N_HC ? c[s] : 0.0f;
        #pragma unroll
        for (int off = 1; off < N_HC; off <<= 1) {
            sum += __shfl_xor_sync(full, sum, off);
        }
        if (tid < N_HC) {
            c[s] *= 1.0f / (sum + eps);
        }
    }
    for (int it = 1; it < sinkhorn_iters; ++it) {
        if (tid < N_HC) {
            float sum = 0.0f;
            #pragma unroll
            for (int s = 0; s < N_HC; ++s) sum += c[s];
            const float inv_denom = 1.0f / (sum + eps);
            #pragma unroll
            for (int s = 0; s < N_HC; ++s) c[s] *= inv_denom;
        }
        #pragma unroll
        for (int s = 0; s < N_HC; ++s) {
            float sum = tid < N_HC ? c[s] : 0.0f;
            #pragma unroll
            for (int off = 1; off < N_HC; off <<= 1) {
                sum += __shfl_xor_sync(full, sum, off);
            }
            if (tid < N_HC) {
                c[s] *= 1.0f / (sum + eps);
            }
        }
    }
    if (tid < N_HC) {
        #pragma unroll
        for (int s = 0; s < N_HC; ++s) {
            row_out[2 * N_HC + tid * N_HC + s] = c[s];
        }
    }
}

static __global__ void dsv4_hc_split_sinkhorn_f32(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float       * __restrict__ dst,
        const int   n_hc,
        const int   sinkhorn_iters,
        const int   n_rows,
        const int   mix_hc,
        const int   nb01,    // input  row stride in bytes
        const int   nb1,     // output row stride in bytes
        const float eps) {
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }

    const int tid   = threadIdx.x;
    const int blksz = blockDim.x;

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    const float * row_in  = (const float *) ((const char *) mixes + row * nb01);
    float       * row_out = (float *)       ((char *)       dst   + row * nb1);

    // ---------------- Section 1: pre slice ----------------
    // out[i] = sigmoid(mix[i] * pre_scale + base[i]) + eps
    for (int i = tid; i < n_hc; i += blksz) {
        const float z = row_in[i] * pre_scale + base[i];
        row_out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    // ---------------- Section 2: post slice ----------------
    // out[n_hc + i] = 2 * sigmoid(mix[n_hc + i] * post_scale + base[n_hc + i])
    for (int i = tid; i < n_hc; i += blksz) {
        const int off = n_hc + i;
        const float z = row_in[off] * post_scale + base[off];
        row_out[off] = 2.0f / (1.0f + expf(-z));
    }

    // ---------------- Section 3: comb matrix Sinkhorn ----------------
    //
    // c[src_hc + dst_hc * n_hc] layout (matches CPU reference at
    // ggml-cpu/ops.cpp:11055). Generic fallback path (shared memory +
    // __syncthreads); the common n_hc == 4 case is served by the
    // register-resident template above.
    extern __shared__ float shmem[];
    float * c = shmem;            // n_hc * n_hc floats

    // Load the comb logits = mix * comb_scale + base (parallel over the block).
    for (int i = tid; i < n_hc * n_hc; i += blksz) {
        const int off = 2 * n_hc + i;
        c[i] = row_in[off] * comb_scale + base[off];
    }
    __syncthreads();

    const int lane = tid;

    // First pass: per-dst_hc softmax (max-subtract, exp, normalize) + eps.
    if (lane < n_hc) {
        const int dst_hc = lane;
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            row_max = fmaxf(row_max, c[src_hc + dst_hc * n_hc]);
        }
        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            c[idx] = c[idx] * inv_sum + eps;
        }
    }
    __syncthreads();

    // First column-normalize: per src_hc, divide by (column sum + eps).
    if (lane < n_hc) {
        const int src_hc = lane;
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            sum += c[src_hc + dst_hc * n_hc];
        }
        const float inv_denom = 1.0f / (sum + eps);
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            c[src_hc + dst_hc * n_hc] *= inv_denom;
        }
    }
    __syncthreads();

    // Remaining sinkhorn_iters - 1 alternations: row-normalize then column-normalize.
    for (int it = 1; it < sinkhorn_iters; ++it) {
        // Row-normalize: per dst_hc, divide by (row sum + eps).
        if (lane < n_hc) {
            const int dst_hc = lane;
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }
        __syncthreads();
        // Column-normalize: per src_hc, divide by (column sum + eps).
        if (lane < n_hc) {
            const int src_hc = lane;
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }
        __syncthreads();
    }

    // Copy the comb matrix back to dst (parallel over the block).
    for (int i = tid; i < n_hc * n_hc; i += blksz) {
        row_out[2 * n_hc + i] = c[i];
    }

    // Suppress unused-warning for mix_hc; it's covered by the host-side asserts.
    (void) mix_hc;
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(mixes->nb[0] == sizeof(float));
    GGML_ASSERT(scale->nb[0] == sizeof(float));
    GGML_ASSERT(base->nb[0]  == sizeof(float));
    GGML_ASSERT(dst->nb[0]   == sizeof(float));

    const int   n_hc           = ggml_get_op_params_i32(dst, 0);
    const int   sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps            = ggml_get_op_params_f32(dst, 2);

    GGML_ASSERT(n_hc > 0 && n_hc <= DSV4_HC_SINKHORN_MAX_N_HC);
    GGML_ASSERT(sinkhorn_iters > 0);

    const int n_rows = (int) ggml_nrows(mixes);
    const int mix_hc = (int) mixes->ne[0];
    const int nb01   = (int) mixes->nb[1];
    const int nb1    = (int) dst->nb[1];

    GGML_ASSERT(mix_hc == (2 + n_hc) * n_hc);
    GGML_ASSERT((int) ggml_nrows(dst) == n_rows);

    // Block size MUST be a warp multiple (>= 32) so that the in-block
    // __syncthreads() barriers are well-formed and any future warp-wide
    // shuffle has a complete mask. With mix_hc in {24, 80} the natural
    // size is rounded up to 32 or 96.
    constexpr int CUDA_WARP_SIZE = 32;
    constexpr int CUDA_MAX_BLOCK = 256;
    const int rounded = ((mix_hc + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE) * CUDA_WARP_SIZE;
    const int threads_per_block = std::min(CUDA_MAX_BLOCK, std::max(CUDA_WARP_SIZE, rounded));

    const dim3 grid(n_rows);
    const dim3 block(threads_per_block);
    const size_t shared = (size_t) n_hc * (size_t) n_hc * sizeof(float);

    cudaStream_t stream = ctx.stream();
    if (n_hc == 4 && threads_per_block == 32) {
        // register-resident fast path (compile-time n_hc, no local-memory spill)
        dsv4_hc_split_sinkhorn_fast<4><<<grid, block, 0, stream>>>(
            (const float *) mixes->data,
            (const float *) scale->data,
            (const float *) base->data,
            (float *)       dst->data,
            sinkhorn_iters, n_rows,
            nb01, nb1, eps);
    } else {
        dsv4_hc_split_sinkhorn_f32<<<grid, block, shared, stream>>>(
            (const float *) mixes->data,
            (const float *) scale->data,
            (const float *) base->data,
            (float *)       dst->data,
            n_hc, sinkhorn_iters, n_rows, mix_hc,
            nb01, nb1, eps);
    }
    CUDA_CHECK(cudaGetLastError());
}
