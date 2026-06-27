#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation (occurrence-based).
        //
        // Each (token, expert-slot) routed to this block's expert gets its OWN compact entry.
        // This matters because some models (e.g. this REAP-pruned DeepSeek-V4) route a token to
        // the same expert in more than one of its top-k slots. The original code used
        // warp_reduce_any (token-presence), so duplicate slots for one token collapsed into a
        // single entry -> ne_get_rows was left partially unwritten -> holes in ids_src1/ids_dst
        // -> out-of-bounds reads in the downstream quantize_mmq / MMQ kernels (illegal access).
        // Requires n_expert_used <= warp_size so each lane inspects at most one slot (true for
        // every supported expert count; the optimized templates only go up to 32 == warp_size).
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The slot index at which this lane's expert matches, if any.
            int match    = 0;  // 1 if this lane's slot routes to this block's expert.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                    match    = 1;
                }
            }

            // Inclusive prefix sum of match over the warp: a matching lane's (prefix-1) is its
            // distinct 0-based position among this token's matches; the last lane holds the total.
            int prefix = match;
#pragma unroll
            for (int offset = 1; offset < warp_size; offset <<= 1) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, prefix, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    prefix += tmp;
                }
            }
            const int n_match_token = __shfl_sync(0xFFFFFFFF, prefix, warp_size - 1, warp_size);

            if (iex_used != -1) {
                store[it_compact + prefix - 1] = mm_ids_helper_store(it, iex_used);
            }

            it_compact += n_match_token;
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        static_assert(n_expert_used == 6 || warp_size % n_expert_used == 0, "bad n_expert_used");
        const int neu_padded = n_expert_used == 6 ? 8 : n_expert_used; // Padded to next higher power of 2.
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;
        ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    // The generic occurrence-based path inspects exactly one expert slot per lane, so it is only
    // correct when n_expert_used <= warp_size (otherwise a lane could own multiple matching slots
    // but only records one). True for every supported expert count (the templates topped out at 32).
    GGML_ASSERT(n_expert_used_var <= warp_size && "mm_ids_helper assumes n_expert_used <= warp_size");
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    // Size the per-block compaction buffer for the worst case of the occurrence-based generic path:
    // a single expert can receive up to n_tokens*n_expert_used routed slots (e.g. if many tokens
    // route duplicate slots to the same expert), so it_compact can exceed n_tokens. (The original
    // presence-based code bounded it_compact by n_tokens; occurrence-based does not.)
    const size_t nbytes_shared = (size_t) n_tokens * n_expert_used_var * sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream) {
    // NOTE: always use the generic (occurrence-based) implementation. The templated optimized
    // paths (case 2/4/6/8/16/32) compact by token-PRESENCE (warp_reduce_any) and therefore
    // collapse a token's duplicate expert slots into one entry, leaving holes in ids_src1/ids_dst
    // for models that route a token to the same expert twice (this REAP DeepSeek-V4 does). The
    // generic path is occurrence-correct; the helper cost is negligible next to the expert
    // matmuls, so routing everything through it is the safe choice. See mm_ids_helper<0> above.
    launch_mm_ids_helper<0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
}
