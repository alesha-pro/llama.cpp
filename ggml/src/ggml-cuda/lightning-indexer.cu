#include "common.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

#include <type_traits>

typedef union {
    int2 i2;
    half2 h2[2];
} half4;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

#include <mma.h>
namespace wmma = nvcuda::wmma;

template <int64_t n_embd, int64_t n_head, ggml_type type_K, bool acc_f16 = false>
static __global__ void lightning_indexer_kernel_wmma(
        const float * src0, const char * src1, const float * src2, float * dst,
        const float scale_embd, const float scale_heads,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb21, size_t nb22, size_t nb23,
        const int skip_pos0 = -1, const int skip_ratio = 0
    ) {

    constexpr int K_VECS_PER_BLOCK = 32;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEADS_PER_INNER_LOOP = 8;
    constexpr int K_EMBD_PER_INNER_LOOP = 16;
    constexpr int n_embd_padded = n_embd + 8;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each block processes K_VECS_PER_BLOCK K vectors
    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;

    // causal skip (DSV4_IDX_SKIP): if every KV row of this block lies beyond
    // the query's causal horizon, write zeros (the graph adds the causal mask
    // right after, turning them into -INF exactly as without the skip) and
    // exit before touching K. n_visible matches fill_compress_causal.
    if (skip_ratio > 0) {
        const int n_visible = (skip_pos0 + i_batch + 1) / skip_ratio;
        if (start_kv >= n_visible) {
            if (tid < K_VECS_PER_BLOCK) {
                const int i_kv = start_kv + tid;
                if (i_kv < n_kv) {
                    float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
                    dst_base[i_kv] = 0.0f;
                }
            }
            return;
        }
    }

    const char  * q_base = (const char  *)                 src0 + i_batch*nb02 + i_stream*nb03;
    const float * w_base = (const float *) ((const char *) src2 + i_batch*nb21 + i_stream*nb23);

    // phase 1 - load weights and first Q tile to shared memory

    __shared__ float w_shared[n_head];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][n_embd_padded / 4];

    if (tid < n_head) {
        w_shared[tid] = w_base[tid];
    }

    // total number of half4 elements in HEADS_PER_INNER_LOOP x n_embd Q tile
    constexpr int n_q_tile = HEADS_PER_INNER_LOOP * (n_embd / 4);
    // number of registers needed in each thread to store Q tile in thread block
    constexpr int n_q_next = (n_q_tile + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    #pragma unroll
    for (int i_q = tid; i_q < n_q_tile; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (n_embd / 4);
        const int i_embd = i_q % (n_embd / 4);
        const float4 q = *(const float4 *) (q_base + i_head*nb01 + i_embd*sizeof(float4));
        half4 q_packed;
        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
        q_shared_h[i_head][i_embd] = q_packed.i2;
    }

    // phase 2 - load (and dequantize if needed) K to shared mem

    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][n_embd_padded / 4][2];

    constexpr int n_k = K_VECS_PER_BLOCK * (n_embd / 4);

    if constexpr (type_K == GGML_TYPE_F16) {
        #pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (n_embd / 4);
            const int i_embd = i_k % (n_embd / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<type_K, half, 4>();
        #pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (n_embd / 4);
            const int i_embd = i_k % (n_embd / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    }

    __syncthreads();

    // phase 3 - calculate lightning indexer scores

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    // load K fragment
    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], n_embd_padded);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < n_head; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        // we don't use accumulator for anything, fill it with zeros
        // acc_f16: each mma only accumulates a 16-wide k-slice, the cross-warp
        // reduction below stays fp32, so the precision loss is bounded
        using acc_t = typename std::conditional<acc_f16, half, float>::type;
        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, acc_t> frag_acc;
        wmma::fill_fragment(frag_acc, (acc_t) 0.0f);

        // load Q fragment
        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], n_embd_padded);

        // preload next Q tile to registers during matrix multiplication
        float4 q_next[n_q_next];

        if (i_head_next < n_head) {
            #pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < n_q_tile; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (n_embd / 4);
                const int i_embd =               i_q % (n_embd / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nb01 + i_embd*sizeof(float4));
            }
        }

        // perform matrix multiplication
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        // each warp stores into its own disjoint qk_shared slab, reinterpreted
        // as half when acc_f16 (the half matrix occupies the slab's first half)
        wmma::store_matrix_sync((acc_t*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        // make sure all threads finished using q_shared_h so we can store next tile
        __syncthreads();

        // write preloaded Q tile to shared memory
        if (i_head_next < n_head) {
            #pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < n_q_tile; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_q / (n_embd / 4);
                const int i_embd = i_q % (n_embd / 4);
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
                q_shared_h[i_head][i_embd] = q_packed.i2;
                ++i_q_next;
            }
        }

        // accumulate QK multiplication results from all block warps
        // (there are 256 threads in block and 256 matmul outputs)
        // TODO it will break if WARP_SIZE is not 32
        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            if constexpr (acc_f16) {
                sum += __half2float(((const half *) &qk_shared[w][0][0])[h*K_VECS_PER_BLOCK + k]);
            } else {
                sum += qk_shared[w][h][k];
            }
        }

        // scale_embd, ReLU, weight
        sum *= scale_embd;
        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        // wait until qk_shared[0] is no longer used
        __syncthreads();

        // reuse qk_shared[0] for storing partial results
        qk_shared[0][h][k] = sum;

        // wait until all threads write their results
        __syncthreads();

        // accumulate result over heads
        if (tid < K_VECS_PER_BLOCK) {
            #pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        // make sure all threads finished using qk_shared
        __syncthreads();
    }

    // phase 4 - store output to VRAM

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = start_kv + tid;
        if (i_kv < n_kv) {
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k * scale_heads;
        }
    }
}

// ------------------------------------------------------------------------
// Q-tiled variant (DSV4_IDX_QTILE=1): one block scores a tile of 16 query
// tokens against 32 KV rows. K is loaded to shared memory ONCE per tile
// instead of once per query token - at deep contexts the compressed-K
// no longer fits in L2 and the per-query reload of the base kernel becomes
// the bottleneck. 8 warps process 8 heads per pass (8 passes for 64 heads):
//   per warp:  KQ = Q_tile[16 x 128] x K^T[128 x 32] via wmma m16n16k16,
//   ReLU + per-(token,head) weight applied on the warp's own smem slab,
//   accumulated in registers, cross-warp (= cross-head) reduction at the end.
// Shared memory exceeds the 48 KB static limit -> dynamic smem + host opt-in.

template <int64_t n_embd, int64_t n_head, ggml_type type_K>
static __global__ void lightning_indexer_kernel_wmma_qtile(
        const float * src0, const char * src1, const float * src2, float * dst,
        const float scale_embd, const float scale_heads,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb21, size_t nb22, size_t nb23,
        const int skip_pos0, const int skip_ratio
    ) {

    constexpr int Q_TILE            = 16;
    constexpr int K_VECS            = 32;
    constexpr int WARPS_PER_BLOCK   = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEAD_PASSES       = n_head / WARPS_PER_BLOCK;
    constexpr int n_embd_padded     = n_embd + 8;
    constexpr int SCORE_LD          = K_VECS + 2; // fp32 slab stride, avoids bank conflicts

    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    const int start_kv = blockIdx.x * K_VECS;
    const int q0       = blockIdx.y * Q_TILE;

    // dynamic smem layout:
    //   k_shared     [K_VECS][n_embd_padded]            half
    //   q_shared     [WARPS][Q_TILE][n_embd_padded]     half
    //   score_shared [WARPS][Q_TILE][SCORE_LD]          float
    //   w_shared     [Q_TILE][n_head]                   float
    extern __shared__ char qtile_smem[];
    half  * k_shared     = (half  *)  qtile_smem;
    half  * q_shared     = (half  *) (qtile_smem + K_VECS*n_embd_padded*sizeof(half));
    float * score_shared = (float *) (qtile_smem + (K_VECS + WARPS_PER_BLOCK*Q_TILE)*n_embd_padded*sizeof(half));
    float * w_shared     = score_shared + WARPS_PER_BLOCK*Q_TILE*SCORE_LD;

    // causal skip for the whole tile: judge by the tile's DEEPEST query -
    // if even it cannot see this KV block, nobody in the tile can
    if (skip_ratio > 0) {
        const int t_last    = min((int) n_batch - 1, q0 + Q_TILE - 1);
        const int n_visible = (skip_pos0 + t_last + 1) / skip_ratio;
        if (start_kv >= n_visible) {
            #pragma unroll
            for (int e = tid; e < Q_TILE*K_VECS; e += THREADS_PER_BLOCK) {
                const int t    = q0 + e / K_VECS;
                const int i_kv = start_kv + e % K_VECS;
                if (t < n_batch && i_kv < n_kv) {
                    float * dst_base = (float *) ((char *) dst + (int64_t) t*nb1 + i_stream*nb3);
                    dst_base[i_kv] = 0.0f;
                }
            }
            return;
        }
    }

    // phase 1: load K tile (dequantize if needed) and the per-(token,head)
    // weights; zero-fill out-of-range rows
    if constexpr (type_K == GGML_TYPE_F16) {
        constexpr int n_k = K_VECS * (n_embd / 4);
        #pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (n_embd / 4);
            const int i_embd  = i_k % (n_embd / 4);
            const int i_kv    = start_kv + i_k_vec;
            int2 * dst4 = (int2 *) &k_shared[i_k_vec*n_embd_padded + i_embd*4];
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                *dst4 = k_base[i_embd];
            } else {
                *dst4 = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<type_K, half, 4>();
        constexpr int n_k = K_VECS * (n_embd / 4);
        #pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (n_embd / 4);
            const int i_embd  = i_k % (n_embd / 4);
            const int i_kv    = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                dequantize_k(k_base, &k_shared[i_k_vec*n_embd_padded + i_embd*4], i_embd * 4);
            } else {
                *(int2 *) &k_shared[i_k_vec*n_embd_padded + i_embd*4] = make_int2(0, 0);
            }
        }
    }

    #pragma unroll
    for (int e = tid; e < Q_TILE*n_head; e += THREADS_PER_BLOCK) {
        const int t = e / n_head;
        const int h = e % n_head;
        const int i_batch = q0 + t;
        w_shared[t*n_head + h] = i_batch < n_batch
            ? ((const float *) ((const char *) src2 + (int64_t) i_batch*nb21 + i_stream*nb23))[h]
            : 0.0f;
    }

    // per-thread accumulator: thread (warp, lane) owns column j=i_lane of all
    // 16 rows of its warp's score slab, summed over the warp's heads
    float acc[Q_TILE] = {0.0f};

    for (int pass = 0; pass < HEAD_PASSES; ++pass) {
        const int i_head = pass*WARPS_PER_BLOCK + i_warp;

        // phase 2: load this pass's Q rows (fp32 -> half), one head per warp
        __syncthreads(); // q_shared reuse across passes
        constexpr int n_q = WARPS_PER_BLOCK * Q_TILE * (n_embd / 4);
        #pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int w_idx  = i_q / (Q_TILE * (n_embd / 4));
            const int rem    = i_q % (Q_TILE * (n_embd / 4));
            const int t      = rem / (n_embd / 4);
            const int i_embd = rem % (n_embd / 4);
            const int i_batch = q0 + t;
            const int h       = pass*WARPS_PER_BLOCK + w_idx;
            half4 q_packed;
            if (i_batch < n_batch) {
                const float4 q = *(const float4 *) ((const char *) src0 + (int64_t) i_batch*nb02 + i_stream*nb03 + h*nb01 + i_embd*sizeof(float4));
                q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
            } else {
                q_packed.i2 = make_int2(0, 0);
            }
            *(int2 *) &q_shared[(w_idx*Q_TILE + t)*n_embd_padded + i_embd*4] = q_packed.i2;
        }
        __syncthreads();

        // phase 3: KQ for this head via wmma, k-dim in steps of 16
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c0, frag_c1;
        wmma::fill_fragment(frag_c0, 0.0f);
        wmma::fill_fragment(frag_c1, 0.0f);
        #pragma unroll
        for (int k0 = 0; k0 < n_embd; k0 += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
            wmma::load_matrix_sync(frag_a, &q_shared[i_warp*Q_TILE*n_embd_padded + k0], n_embd_padded);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
            wmma::load_matrix_sync(frag_b, &k_shared[0*n_embd_padded + k0], n_embd_padded);
            wmma::mma_sync(frag_c0, frag_a, frag_b, frag_c0);
            wmma::load_matrix_sync(frag_b, &k_shared[16*n_embd_padded + k0], n_embd_padded);
            wmma::mma_sync(frag_c1, frag_a, frag_b, frag_c1);
        }
        float * my_slab = &score_shared[i_warp*Q_TILE*SCORE_LD];
        wmma::store_matrix_sync(my_slab,      frag_c0, SCORE_LD, wmma::mem_row_major);
        wmma::store_matrix_sync(my_slab + 16, frag_c1, SCORE_LD, wmma::mem_row_major);
        __syncwarp();

        // phase 4: ReLU + weight on the warp's own slab, accumulate in regs
        #pragma unroll
        for (int t = 0; t < Q_TILE; ++t) {
            float v = my_slab[t*SCORE_LD + i_lane];
            v *= scale_embd;
            v  = v > 0.0f ? v : 0.0f;
            acc[t] += v * w_shared[t*n_head + i_head];
        }
    }

    // phase 5: cross-warp (= cross-head-group) reduction via the score slabs
    __syncthreads();
    float * my_slab = &score_shared[i_warp*Q_TILE*SCORE_LD];
    #pragma unroll
    for (int t = 0; t < Q_TILE; ++t) {
        my_slab[t*SCORE_LD + i_lane] = acc[t];
    }
    __syncthreads();

    #pragma unroll
    for (int e = tid; e < Q_TILE*K_VECS; e += THREADS_PER_BLOCK) {
        const int t = e / K_VECS;
        const int j = e % K_VECS;
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += score_shared[(w*Q_TILE + t)*SCORE_LD + j];
        }
        const int i_batch = q0 + t;
        const int i_kv    = start_kv + j;
        if (i_batch < n_batch && i_kv < n_kv) {
            float * dst_base = (float *) ((char *) dst + (int64_t) i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = sum * scale_heads;
        }
    }
}

#else // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

template <int64_t n_embd, int64_t n_head, ggml_type type_K, bool acc_f16 = false>
static __global__ void lightning_indexer_kernel_wmma(
        const float * src0, const char * src1, const float * src2, float * dst,
        const float scale_embd, const float scale_heads,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb21, size_t nb22, size_t nb23,
        const int skip_pos0 = -1, const int skip_ratio = 0
    ) {
    GGML_UNUSED_VARS(src0, src1, src2, dst,
        scale_embd, scale_heads,
        n_stream, n_batch, n_kv,
        nb1, nb2, nb3,
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        nb21, nb22, nb23, skip_pos0, skip_ratio);
    NO_DEVICE_CODE;
}

template <int64_t n_embd, int64_t n_head, ggml_type type_K>
static __global__ void lightning_indexer_kernel_wmma_qtile(
        const float * src0, const char * src1, const float * src2, float * dst,
        const float scale_embd, const float scale_heads,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb21, size_t nb22, size_t nb23,
        const int skip_pos0, const int skip_ratio
    ) {
    GGML_UNUSED_VARS(src0, src1, src2, dst,
        scale_embd, scale_heads,
        n_stream, n_batch, n_kv,
        nb1, nb2, nb3,
        nb01, nb02, nb03,
        nb11, nb12, nb13,
        nb21, nb22, nb23, skip_pos0, skip_ratio);
    NO_DEVICE_CODE;
}

#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 or half4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (n_embd)

template <int64_t n_embd, int64_t n_head, ggml_type type_K>
static __global__ void lightning_indexer_kernel_vec(
        const float * src0, const char * src1, const float * src2, float * dst,
        const float scale_embd, const float scale_heads,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nb01, size_t nb02, size_t nb03,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb21, size_t nb22, size_t nb23,
        const int skip_pos0 = -1, const int skip_ratio = 0
    ) {

    constexpr int K_VECS_PER_WARP = 8;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * (WARPS_PER_BLOCK * K_VECS_PER_WARP);
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    // causal skip (DSV4_IDX_SKIP): whole-block early-out, see the wmma kernel
    if (skip_ratio > 0) {
        const int n_visible = (skip_pos0 + i_batch + 1) / skip_ratio;
        if (start_kv_block >= n_visible) {
            if (tid < WARPS_PER_BLOCK * K_VECS_PER_WARP) {
                const int i_kv = start_kv_block + tid;
                if (i_kv < n_kv) {
                    float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
                    dst_base[i_kv] = 0.0f;
                }
            }
            return;
        }
    }

    const char  * q_base = (const char  *)                 src0 + i_batch*nb02 + i_stream*nb03;
    const float * w_base = (const float *) ((const char *) src2 + i_batch*nb21 + i_stream*nb23);

    // phase 1 - load (and dequantize if needed) K to registers

    // K are loaded to registers either as float4 or half2 depending on K type
    float4 k_reg_f[K_VECS_PER_WARP];
    half2  k_reg_h[K_VECS_PER_WARP][2];

    if constexpr (type_K == GGML_TYPE_F32) {
        // direct copy of float4
        #pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const float4 * k_base = (const float4 *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        // dequantize bf16 to float
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<type_K, float, 4>();
        #pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else if constexpr (type_K == GGML_TYPE_F16) {
        // direct copy of halfs - separate case for some extra performance
        #pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                *(int2*) &k_reg_h[k] = k_base[i_lane];
            } else {
                *(int2*) &k_reg_h[k] = make_int2(0, 0);
            }
        }
    } else {
        // dequantize remaining types to half
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<type_K, half, 4>();
        #pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) src1 + i_kv*nb12 + i_stream*nb13);
                dequantize_k(k_base, &k_reg_h[k][0], i_lane * 4);
            } else {
                *(int2*) &k_reg_h[k] = make_int2(0, 0);
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    // load weights and Q only for n_head_inner heads at once to reduce shared memory usage
    constexpr int n_head_inner = n_head / 4;

    for (int i_head_0 = 0; i_head_0 < n_head; i_head_0 += n_head_inner) {
        // phase 2 - load weights and Q to shared memory

        __shared__ float  w_shared[n_head_inner];
        // Q are loaded to shared memory either as float4 or half4 (stored as int2) depending on K type
        __shared__ float4 q_shared_f[n_head_inner][n_embd / 4];
        __shared__ int2   q_shared_h[n_head_inner][n_embd / 4];

        if (tid < n_head_inner) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = n_head_inner * (n_embd / 4);
        #pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (n_embd / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (n_embd / 4);
            if constexpr (type_K == GGML_TYPE_F32 || type_K == GGML_TYPE_BF16) {
                q_shared_f[i_head_inner][i_embd] = *(const float4 *) (q_base + i_head*nb01 + i_embd*sizeof(float4));
            } else {
                const float4 q = *(const float4 *) (q_base + i_head*nb01 + i_embd*sizeof(float4));
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
                q_shared_h[i_head_inner][i_embd] = q_packed.i2;
            }
        }

        __syncthreads();

        // phase 3 - calculate lightning indexer scores

        for (int i_head_inner = 0; i_head_inner < n_head_inner; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            if constexpr (type_K == GGML_TYPE_F32 || type_K == GGML_TYPE_BF16) {
                // dot product of floats for f32 and bf16
                const float4 q_vec = q_shared_f[i_head_inner][i_lane];

                #pragma unroll
                for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                    ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                    ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                    ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                    ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
                }
            } else {
                // dot product of halfs for remaining types
                half4 q_vec;
                q_vec.i2 = q_shared_h[i_head_inner][i_lane];
                const half2 q_h0 = q_vec.h2[0];
                const half2 q_h1 = q_vec.h2[1];

                #pragma unroll
                for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                    ggml_cuda_mad(qk[k], q_h0.x, k_reg_h[k][0].x);
                    ggml_cuda_mad(qk[k], q_h0.y, k_reg_h[k][0].y);
                    ggml_cuda_mad(qk[k], q_h1.x, k_reg_h[k][1].x);
                    ggml_cuda_mad(qk[k], q_h1.y, k_reg_h[k][1].y);
                }
            }

            #pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                // scale_embd, ReLU, weight
                if (i_lane == 0) {
                    sum *= scale_embd;
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[WARPS_PER_BLOCK * K_VECS_PER_WARP];

    if (i_lane == 0) {
        #pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k] * scale_heads;
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

    if (tid < WARPS_PER_BLOCK * K_VECS_PER_WARP) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid];
        }
    }
}

#define DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, type_K) \
    template __global__ void lightning_indexer_kernel<n_embd, n_head, type_K>(        \
        const float * src0, const char * src1, const float * src2, float * dst,       \
        const float scale_embd, const float scale_heads,                              \
        int64_t n_stream, int64_t n_batch, int64_t n_kv,                              \
        size_t nb1, size_t nb2, size_t nb3,                                           \
        size_t nb01, size_t nb02, size_t nb03,                                        \
        size_t nb11, size_t nb12, size_t nb13,                                        \
        size_t nb21, size_t nb22, size_t nb23,                                        \
        const int skip_pos0, const int skip_ratio);

#define DECL_LIGHTNING_INDEXER_CASE_ACC(n_embd, n_head, type_K, acc_f16)                    \
    template __global__ void lightning_indexer_kernel_wmma<n_embd, n_head, type_K, acc_f16>( \
        const float * src0, const char * src1, const float * src2, float * dst,             \
        const float scale_embd, const float scale_heads,                                    \
        int64_t n_stream, int64_t n_batch, int64_t n_kv,                                    \
        size_t nb1, size_t nb2, size_t nb3,                                                 \
        size_t nb01, size_t nb02, size_t nb03,                                              \
        size_t nb11, size_t nb12, size_t nb13,                                              \
        size_t nb21, size_t nb22, size_t nb23,                                              \
        const int skip_pos0, const int skip_ratio);

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_F16,  false)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_Q4_0, false)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_Q4_1, false)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_Q5_0, false)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_Q5_1, false)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_Q8_0, false)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_F16,  true)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_Q4_0, true)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_Q4_1, true)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_Q5_0, true)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_Q5_1, true)
DECL_LIGHTNING_INDEXER_CASE_ACC(128, 64, GGML_TYPE_Q8_0, true)

DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma_qtile, 128, 64, GGML_TYPE_F16)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma_qtile, 128, 64, GGML_TYPE_Q4_0)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma_qtile, 128, 64, GGML_TYPE_Q4_1)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma_qtile, 128, 64, GGML_TYPE_Q5_0)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma_qtile, 128, 64, GGML_TYPE_Q5_1)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma_qtile, 128, 64, GGML_TYPE_Q8_0)
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_F16)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_Q4_0)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_Q4_1)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_Q5_0)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_Q5_1)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_Q8_0)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_BF16)
DECL_LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, GGML_TYPE_F32)

#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)         \
    if (K->type == (type_K)) {                                                              \
        lightning_indexer_kernel<n_embd, n_head, type_K><<<grid, block, 0, ctx.stream()>>>( \
            src0_d, src1_d, src2_d, dst_d, scale_embd, scale_heads,                         \
            n_stream, n_batch, n_kv,                                                        \
            nb1, nb2, nb3,                                                                  \
            nb01, nb02, nb03,                                                               \
            nb11, nb12, nb13,                                                               \
            nb21, nb22, nb23,                                                               \
            skip_pos0, skip_ratio                                                           \
        );                                                                                  \
    } else

void ggml_cuda_op_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float scale_embd = ggml_get_op_params_f32(dst, 0);
    const float scale_heads = ggml_get_op_params_f32(dst, 1);

    // optional causal skip (DSV4_IDX_SKIP, set by the graph builder in
    // op_params[2..4]); all-zero op_params (any other caller) disable it
    const int32_t * opi   = (const int32_t *) dst->op_params;
    const int skip_pos0   = opi[4] != 0 ? opi[2] : -1;
    const int skip_ratio  = opi[4] != 0 ? opi[3] : 0;

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);

    GGML_TENSOR_TERNARY_OP_LOCALS

    // input tensor rows must be contiguous
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));
    GGML_ASSERT(nb20 == ggml_type_size(src2->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = src0->ne[0];
    const int n_head   = src0->ne[1];
    const int n_batch  = src0->ne[2];
    const int n_stream = src0->ne[3];
    const int n_kv     = src1->ne[2];

    const float * src0_d = (const float *) src0->data;
    const char  * src1_d = (const char  *) src1->data;
    const float * src2_d = (const float *) src2->data;
    float       * dst_d  = (float *)       dst->data;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;

    if (n_embd == 128 && n_head == 64) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && ampere_mma_available(cc) && src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            // opt-in Q-tiled kernel (DSV4_IDX_QTILE=1): 16 queries per block,
            // K loaded to smem once per tile; needs >48KB smem -> dynamic
            static const bool idx_qtile = getenv("DSV4_IDX_QTILE") != nullptr;
            if (idx_qtile && n_batch >= 16) {
                static bool logged_qtile = false;
                if (!logged_qtile) {
                    fprintf(stderr, "%s: DSV4_IDX_QTILE=1 - lightning indexer using q-tiled kernel\n", __func__);
                    logged_qtile = true;
                }
                constexpr int Q_TILE = 16;
                constexpr int QT_K_VECS = 32;
                constexpr int QT_WARPS = 8;
                constexpr size_t qt_smem =
                    (QT_K_VECS + QT_WARPS*Q_TILE) * (128 + 8) * sizeof(half) +
                    QT_WARPS*Q_TILE*(QT_K_VECS + 2) * sizeof(float) +
                    Q_TILE*64 * sizeof(float);
                dim3 qt_block(32, QT_WARPS);
                dim3 qt_grid((n_kv + QT_K_VECS - 1) / QT_K_VECS, (n_batch + Q_TILE - 1) / Q_TILE, n_stream);
                #define LIGHTNING_INDEXER_CASE_QTILE(n_embd, n_head, K, type_K)                                \
                    if (K->type == (type_K)) {                                                                 \
                        /* smem opt-in is PER DEVICE - a single static flag */                                 \
                        /* breaks on the 2nd GPU of a --split-mode layer run */                                \
                        static bool attr_set[GGML_CUDA_MAX_DEVICES] = {false};                                 \
                        if (!attr_set[device]) {                                                               \
                            CUDA_CHECK(cudaFuncSetAttribute(                                                  \
                                lightning_indexer_kernel_wmma_qtile<n_embd, n_head, type_K>,                  \
                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int) qt_smem));                 \
                            attr_set[device] = true;                                                          \
                        }                                                                                     \
                        lightning_indexer_kernel_wmma_qtile<n_embd, n_head, type_K>                           \
                            <<<qt_grid, qt_block, qt_smem, ctx.stream()>>>(                                   \
                            src0_d, src1_d, src2_d, dst_d, scale_embd, scale_heads,                           \
                            n_stream, n_batch, n_kv,                                                          \
                            nb1, nb2, nb3, nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23,              \
                            skip_pos0, skip_ratio);                                                           \
                    } else
                LIGHTNING_INDEXER_CASE_QTILE(128, 64, src1, GGML_TYPE_F16)
                LIGHTNING_INDEXER_CASE_QTILE(128, 64, src1, GGML_TYPE_Q4_0)
                LIGHTNING_INDEXER_CASE_QTILE(128, 64, src1, GGML_TYPE_Q4_1)
                LIGHTNING_INDEXER_CASE_QTILE(128, 64, src1, GGML_TYPE_Q5_0)
                LIGHTNING_INDEXER_CASE_QTILE(128, 64, src1, GGML_TYPE_Q5_1)
                LIGHTNING_INDEXER_CASE_QTILE(128, 64, src1, GGML_TYPE_Q8_0)
                GGML_ABORT("fatal error");
            }

            // opt-in fp16 mma accumulator (indexer only ranks blocks; the
            // cross-warp reduction stays fp32)
            static const bool idx_f16acc = getenv("DSV4_IDX_F16ACC") != nullptr;
            if (idx_f16acc) {
                static bool logged = false;
                if (!logged) {
                    fprintf(stderr, "%s: DSV4_IDX_F16ACC=1 - lightning indexer using fp16 mma accumulator\n", __func__);
                    logged = true;
                }
                #define LIGHTNING_INDEXER_CASE_ACC(n_embd, n_head, K, type_K)                                    \
                    if (K->type == (type_K)) {                                                                   \
                        lightning_indexer_kernel_wmma<n_embd, n_head, type_K, true><<<grid, block, 0, ctx.stream()>>>( \
                            src0_d, src1_d, src2_d, dst_d, scale_embd, scale_heads,                              \
                            n_stream, n_batch, n_kv,                                                             \
                            nb1, nb2, nb3, nb01, nb02, nb03, nb11, nb12, nb13, nb21, nb22, nb23,                 \
                            skip_pos0, skip_ratio);                                                              \
                    } else
                LIGHTNING_INDEXER_CASE_ACC(128, 64, src1, GGML_TYPE_F16)
                LIGHTNING_INDEXER_CASE_ACC(128, 64, src1, GGML_TYPE_Q4_0)
                LIGHTNING_INDEXER_CASE_ACC(128, 64, src1, GGML_TYPE_Q4_1)
                LIGHTNING_INDEXER_CASE_ACC(128, 64, src1, GGML_TYPE_Q5_0)
                LIGHTNING_INDEXER_CASE_ACC(128, 64, src1, GGML_TYPE_Q5_1)
                LIGHTNING_INDEXER_CASE_ACC(128, 64, src1, GGML_TYPE_Q8_0)
                GGML_ABORT("fatal error");
            }

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, src1, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, src1, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else {
        GGML_ABORT("fatal error");
    }
}
