#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// Exact single-row Top-512 specialized for long-context sparse attention.
// DeviceTopK is slightly slower than a full radix argsort for K=512 on Ampere
// (both are ~130 us at N=32768). This kernel radix-selects only the cutoff
// value, then compacts indices above/equal to it. Output order is unspecified,
// which is already part of GGML_OP_TOP_K's contract.
static __device__ __forceinline__ unsigned int top_k_ordered_float(float x) {
    const unsigned int u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

// state = { prefix, prefix_mask, n_above, out_gt, out_eq }
static __global__ void top_k_512_radix_hist(
        const float * __restrict__ src,
        int ncols,
        const unsigned int * __restrict__ state,
        unsigned int * __restrict__ hist_out,
        int shift) {
    __shared__ unsigned int hist[256];
    const int tid = threadIdx.x;
    hist[tid] = 0;
    __syncthreads();

    const unsigned int prefix = state[0];
    const unsigned int prefix_mask = state[1];
    for (int i = blockIdx.x * blockDim.x + tid; i < ncols; i += gridDim.x * blockDim.x) {
        const unsigned int key = top_k_ordered_float(src[i]);
        if ((key & prefix_mask) == prefix) {
            atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
        }
    }
    __syncthreads();
    hist_out[blockIdx.x * 256 + tid] = hist[tid];
}

static __global__ void top_k_512_radix_choose(
        const unsigned int * __restrict__ hist,
        unsigned int * __restrict__ state,
        int nblocks,
        int shift) {
    __shared__ unsigned int totals[256];
    const int tid = threadIdx.x;
    unsigned int sum = 0;
    for (int b = 0; b < nblocks; ++b) {
        sum += hist[b * 256 + tid];
    }
    totals[tid] = sum;
    __syncthreads();

    if (tid == 0) {
        constexpr unsigned int K = 512;
        const unsigned int need = K - state[2];
        unsigned int higher = 0;
        for (int b = 255; b >= 0; --b) {
            const unsigned int count = totals[b];
            if (higher + count >= need) {
                state[0] |= (unsigned int) b << shift;
                state[1] |= 0xffu << shift;
                state[2] += higher;
                break;
            }
            higher += count;
        }
    }
}

static __global__ void top_k_512_radix_compact(
        const float * __restrict__ src,
        int * __restrict__ dst,
        int ncols,
        unsigned int * __restrict__ state) {
    constexpr unsigned int K = 512;
    const unsigned int threshold = state[0];
    const unsigned int n_above = state[2];
    const unsigned int n_equal_needed = K - n_above;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ncols; i += gridDim.x * blockDim.x) {
        const unsigned int key = top_k_ordered_float(src[i]);
        if (key > threshold) {
            const unsigned int pos = atomicAdd(&state[3], 1u);
            if (pos < K) dst[pos] = i;
        } else if (key == threshold) {
            const unsigned int pos = atomicAdd(&state[4], 1u);
            if (pos < n_equal_needed) dst[n_above + pos] = i;
        }
    }
}

// Exact batched Top-K by radix-select: one CUDA block per row, all four 8-bit
// passes kept in shared memory.  This exists because the DSV4 *prefill* top-k
// went through GGML_OP_ARGSORT, which materializes a full [ncols, nrows] i32
// sort plus CUB temp storage - an allocation that scales with context and ran
// the 4x3090 rig out of VRAM at ~90K prompt tokens.  Selecting instead of
// sorting needs no context-scaled scratch at all: the only output is the
// [k, nrows] index tensor.
//
// Same contract as the single-row kernel above: the returned set is exact, its
// order is unspecified, and GGML_OP_TOP_K does not promise an order.
static __global__ void top_k_radix_rows(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const int ncols,
        const int k) {
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix;
    __shared__ unsigned int s_mask;
    __shared__ unsigned int s_nabove;
    __shared__ unsigned int s_ctr_gt;
    __shared__ unsigned int s_ctr_eq;

    const int     tid  = threadIdx.x;
    const float * srow = src + (int64_t) blockIdx.x * ncols;
    int         * drow = dst + (int64_t) blockIdx.x * k;

    if (tid == 0) {
        s_prefix = 0u;
        s_mask   = 0u;
        s_nabove = 0u;
        s_ctr_gt = 0u;
        s_ctr_eq = 0u;
    }
    __syncthreads();

    #pragma unroll
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0u;
        __syncthreads();

        const unsigned int prefix = s_prefix;
        const unsigned int pmask  = s_mask;
        for (int i = tid; i < ncols; i += blockDim.x) {
            const unsigned int key = top_k_ordered_float(srow[i]);
            if ((key & pmask) == prefix) {
                atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();

        if (tid == 0) {
            const unsigned int need   = (unsigned int) k - s_nabove;
            unsigned int       higher = 0;
            for (int b = 255; b >= 0; --b) {
                const unsigned int count = hist[b];
                if (higher + count >= need) {
                    s_prefix |= (unsigned int) b << shift;
                    s_mask   |= 0xffu << shift;
                    s_nabove += higher;
                    break;
                }
                higher += count;
            }
        }
        __syncthreads();
    }

    const unsigned int threshold = s_prefix;
    const unsigned int n_above   = s_nabove;
    const unsigned int n_eq_need = (unsigned int) k - n_above;
    for (int i = tid; i < ncols; i += blockDim.x) {
        const unsigned int key = top_k_ordered_float(srow[i]);
        if (key > threshold) {
            const unsigned int pos = atomicAdd(&s_ctr_gt, 1u);
            if (pos < (unsigned int) k) {
                drow[pos] = i;
            }
        } else if (key == threshold) {
            const unsigned int pos = atomicAdd(&s_ctr_eq, 1u);
            if (pos < n_eq_need) {
                drow[n_above + pos] = i;
            }
        }
    }
}

static __global__ void top_k_512_radix_cooperative(
        const float * __restrict__ src,
        int * __restrict__ dst,
        int ncols,
        unsigned int * __restrict__ hist_global,
        unsigned int * __restrict__ state) {
    constexpr unsigned int K = 512;
    cg::grid_group grid = cg::this_grid();
    __shared__ unsigned int hist[256];
    const int tid = threadIdx.x;
    volatile unsigned int * vstate = state;

    if (blockIdx.x == 0 && tid < 5) vstate[tid] = 0;
    __threadfence();
    grid.sync();

    #pragma unroll
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0;
        __syncthreads();

        const unsigned int prefix = vstate[0];
        const unsigned int prefix_mask = vstate[1];
        for (int i = blockIdx.x * blockDim.x + tid; i < ncols; i += gridDim.x * blockDim.x) {
            const unsigned int key = top_k_ordered_float(src[i]);
            if ((key & prefix_mask) == prefix) {
                atomicAdd(&hist[(key >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();
        hist_global[blockIdx.x * 256 + tid] = hist[tid];
        __threadfence();
        grid.sync();

        if (blockIdx.x == 0) {
            unsigned int sum = 0;
            for (int b = 0; b < gridDim.x; ++b) sum += hist_global[b * 256 + tid];
            hist[tid] = sum;
            __syncthreads();
            if (tid == 0) {
                const unsigned int need = K - vstate[2];
                unsigned int higher = 0;
                for (int b = 255; b >= 0; --b) {
                    const unsigned int count = hist[b];
                    if (higher + count >= need) {
                        vstate[0] |= (unsigned int) b << shift;
                        vstate[1] |= 0xffu << shift;
                        vstate[2] += higher;
                        break;
                    }
                    higher += count;
                }
            }
        }
        __threadfence();
        grid.sync();
    }

    const unsigned int threshold = vstate[0];
    const unsigned int n_above = vstate[2];
    const unsigned int n_equal_needed = K - n_above;
    for (int i = blockIdx.x * blockDim.x + tid; i < ncols; i += gridDim.x * blockDim.x) {
        const unsigned int key = top_k_ordered_float(src[i]);
        if (key > threshold) {
            const unsigned int pos = atomicAdd(&state[3], 1u);
            if (pos < K) dst[pos] = i;
        } else if (key == threshold) {
            const unsigned int pos = atomicAdd(&state[4], 1u);
            if (pos < n_equal_needed) dst[n_above + pos] = i;
        }
    }
}

#endif // !GGML_USE_HIP && !GGML_USE_MUSA

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (nrows == 1 && k == 512 && ncols >= 2048) {
        const int nblocks = std::min<int>(128, ((int) ncols + 1023) / 1024);
        ggml_cuda_pool_alloc<unsigned int> hist_alloc(pool, (size_t) nblocks * 256);
        ggml_cuda_pool_alloc<unsigned int> state_alloc(pool, 5);
        unsigned int * hist = hist_alloc.get();
        unsigned int * state = state_alloc.get();
        const float * src_arg = src0_d;
        int * dst_arg = dst_d;
        int ncols_arg = (int) ncols;
        unsigned int * hist_arg = hist;
        unsigned int * state_arg = state;
        void * args[] = { &src_arg, &dst_arg, &ncols_arg, &hist_arg, &state_arg };
        const cudaError_t coop_err = cudaLaunchCooperativeKernel(
                (void *) top_k_512_radix_cooperative,
                dim3(nblocks), dim3(256), args, 0, stream);
        if (coop_err == cudaSuccess) {
            return;
        }
        // Cooperative launch can be unavailable under some graph/runtime
        // combinations. Clear its error and use the validated multi-launch path.
        (void) cudaGetLastError();
        CUDA_CHECK(cudaMemsetAsync(state, 0, 5 * sizeof(unsigned int), stream));
        for (int shift = 24; shift >= 0; shift -= 8) {
            top_k_512_radix_hist<<<nblocks, 256, 0, stream>>>(src0_d, (int) ncols, state, hist, shift);
            top_k_512_radix_choose<<<1, 256, 0, stream>>>(hist, state, nblocks, shift);
        }
        top_k_512_radix_compact<<<nblocks, 256, 0, stream>>>(src0_d, dst_d, (int) ncols, state);
        return;
    }
    // Batched exact select, one block per row. Requires ncols >= k so that the
    // radix invariant (the chosen prefix always holds at least `need` entries)
    // holds; below that the caller has nothing to select anyway.
    if (nrows > 1 && k >= 1 && ncols >= k && ncols >= 512) {
        top_k_radix_rows<<<(unsigned int) nrows, 256, 0, stream>>>(
                src0_d, dst_d, (int) ncols, (int) k);
        return;
    }
#endif
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ncols > 1024) {
        argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
