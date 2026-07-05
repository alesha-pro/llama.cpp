#include "models.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

void llama_model_deepseek4::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,  hparams.n_lora_o);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,hparams.n_attn_out_groups);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);
    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SQRTSOFTPLUS;
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,          hparams.n_swa, false);
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(0, false);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE, hparams.compress_rope_freq_base, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,      hparams.indexer_n_head, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,      hparams.indexer_head_size, false);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,           hparams.indexer_top_k, false);
    ml.get_key(LLM_KV_HASH_LAYER_COUNT,                  hparams.n_hash_layers);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,            hparams.n_hc);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERS,   hparams.hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPS,              hparams.hc_eps);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,           hparams.swiglu_clamp_exp, hparams.n_layer, false);

    std::vector<uint32_t> compress_ratios;
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, compress_ratios);
    if (compress_ratios.size() < hparams.n_layer) {
        throw std::runtime_error(format("DeepSeek V4 compress ratio count mismatch: got %zu, expected %u",
                    compress_ratios.size(), hparams.n_layer));
    }
    std::copy_n(compress_ratios.begin(), hparams.n_layer, hparams.attn_compress_ratio.begin());

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        const uint32_t ratio = hparams.attn_compress_ratio[il];
        if (ratio == 0) {
            continue;
        }

        const uint32_t coff = ratio == 4 ? 2 : 1;
        uint32_t state_size = coff * ratio * coff * hparams.n_embd_head_k(il);
        if (ratio == 4) {
            state_size += coff * ratio * coff * hparams.indexer_head_size;
        }
        hparams.dsv4_state_size = std::max(hparams.dsv4_state_size, state_size);
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_deepseek4::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank       = hparams.n_lora_q;
    const int64_t o_lora_rank       = hparams.n_lora_o;
    const int64_t n_out_groups      = hparams.n_attn_out_groups;
    const int64_t n_ff_exp          = hparams.n_ff_exp;
    const int64_t n_expert_shared   = hparams.n_expert_shared;
    const int64_t n_hc              = hparams.n_hc;
    const int64_t hc_dim            = n_hc * n_embd;
    const int64_t hc_mix            = (2 + n_hc) * n_hc;

    if (n_out_groups == 0) {
        throw std::runtime_error("DeepSeek V4 requires attention output groups");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,     "weight"), {n_embd}, 0);
    output          = create_tensor(tn(LLM_TENSOR_OUTPUT,          "weight"), {n_embd, n_vocab}, 0);
    output_hc_base  = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_BASE,  "weight"), {n_hc}, 0);
    output_hc_fn    = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_FN,    "weight"), {hc_dim, n_hc}, 0);
    output_hc_scale = create_tensor(tn(LLM_TENSOR_OUTPUT_HC_SCALE, "weight"), {1}, 0);

    auto create_deepseek4_compressor = [&](llama_layer & layer, int bid, int64_t compress_ratio, int64_t head_size, bool indexer) {
        const int64_t coff = compress_ratio == 4 ? 2 : 1;
        ggml_tensor *& ape  = indexer ? layer.indexer_compressor_ape  : layer.attn_compressor_ape;
        ggml_tensor *& kv   = indexer ? layer.indexer_compressor_kv   : layer.attn_compressor_kv;
        ggml_tensor *& gate = indexer ? layer.indexer_compressor_gate : layer.attn_compressor_gate;
        ggml_tensor *& norm = indexer ? layer.indexer_compressor_norm : layer.attn_compressor_norm;

        ape  = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_APE  : LLM_TENSOR_ATTN_COMPRESSOR_APE,  "weight", bid), {coff * head_size, compress_ratio}, 0);
        kv   = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_KV   : LLM_TENSOR_ATTN_COMPRESSOR_KV,   "weight", bid), {n_embd, coff * head_size}, 0);
        gate = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_GATE : LLM_TENSOR_ATTN_COMPRESSOR_GATE, "weight", bid), {n_embd, coff * head_size}, 0);
        norm = create_tensor(tn(indexer ? LLM_TENSOR_INDEXER_COMPRESSOR_NORM : LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", bid), {head_size}, 0);
    };

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t compress_ratio = hparams.attn_compress_ratio[i];

        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix}, 0);
        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        layer.attn_sinks     = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,     "weight", i), {n_head}, 0);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, 0);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.wq_a    = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,    "weight", i), {n_embd, q_lora_rank}, 0);
        layer.wq_b    = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,    "weight", i), {q_lora_rank, n_head * n_embd_head_k}, 0);
        layer.attn_kv = create_tensor(tn(LLM_TENSOR_ATTN_KV,     "weight", i), {n_embd, n_embd_head_k}, 0);
        layer.attn_wo_a = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A, "weight", i), {n_head * n_embd_head_v / n_out_groups, n_out_groups * o_lora_rank}, 0);
        layer.attn_wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B, "weight", i), {n_out_groups * o_lora_rank, n_embd}, 0);

        if (compress_ratio > 0) {
            create_deepseek4_compressor(layer, i, compress_ratio, n_embd_head_k, false);
        }
        if (compress_ratio == 4) {
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size}, 0);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, 0);
            create_deepseek4_compressor(layer, i, compress_ratio, hparams.indexer_head_size, true);
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        if (static_cast<uint32_t>(i) < hparams.n_hash_layers) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, 0);
            layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, 0);
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, TENSOR_NOT_REQUIRED);
        }

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,   n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,   n_ff_exp * n_expert_shared}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

namespace {

struct dsv4_hc_mix {
    ggml_tensor * x;
    ggml_tensor * mixes;
    ggml_tensor * pre;
    ggml_tensor * post;
    ggml_tensor * comb;
};

struct dsv4_state_pair {
    ggml_tensor * kv;
    ggml_tensor * score;
};

struct dsv4_decode_compressor {
    ggml_tensor * kv_state;
    ggml_tensor * score_state;
    ggml_tensor * kv_comp;
};

struct dsv4_state_layout {
    int64_t width;
    int64_t rows;
    int64_t elems;
};

enum class dsv4_mask_kind {
    RAW_WINDOW,
    COMPRESS_CAUSAL,
    ATTN_STATIC,
    // Like COMPRESS_CAUSAL but the tensor width (ne[0] = NC_FIXED) is a depth-
    // INVARIANT constant: rows [0, n_comp_visible) get the causal 0, rows
    // [n_comp_visible, NC_FIXED) stay -INF. Its can_reuse never fails on depth,
    // so the decode CUDA graph can REPLAY across compression boundaries instead
    // of recapturing every `ratio` tokens. Pairs with a fixed-width cache view.
    COMPRESS_FIXED,
    // Like COMPRESS_FIXED but the width is BUCKETED instead of constant:
    // ne[0] = min(GGML_PAD(n_comp_visible, bucket), cap), with the bucket size
    // stored in `window` and the cap in `n_comp`. can_reuse recomputes the
    // bucketed width from the new ubatch, so the shape (and the graph) changes
    // only every bucket*ratio tokens instead of every ratio tokens. Used to
    // mask the bucketed indexer-score view in the top-k GATHER decode path
    // (rows beyond n_comp_visible are zero in the cache and would relu-score
    // 0.0, potentially outranking real rows -> mask them to -INF before top-k).
    COMPRESS_BUCKET,
};

struct dsv4_mask_entry {
    ggml_tensor   * tensor = nullptr;
    dsv4_mask_kind kind;
    int64_t         n_raw = 0;
    int64_t         n_comp = 0;
    int64_t         window = 0;
    int64_t         ratio = 0;
};

// A per-token scalar index fed as a graph INPUT (refreshed every token by
// set_input). Used to replace position-derived view offsets that would
// otherwise be baked into the graph at build time (and go stale on graph
// reuse). value = base + (last_pos % ratio).
// Per-token scalar input. `mode` selects the value formula computed in set_input
// (from last_pos and ratio). All are constant-SHAPE [1] inputs so the decode
// graph stays reusable; only their DATA refreshes per token.
enum class dsv4_idx_mode {
    POS_MOD = 0,        // i32: base + last_pos % ratio          (baked view-offset replacement)
    COMP_POS,           // i32: last_pos + 1 - ratio             (RoPE pos of the emitted compressed chunk)
    COMP_ROW,           // i32: boundary ? (last_pos+1)/ratio-1 : base   (cache write row; base = scratch row)
    BOUNDARY_FLAG,      // f32: boundary ? 1.0 : 0.0             (state-shift blend selector)
};

struct dsv4_index_entry {
    ggml_tensor * tensor = nullptr;
    int64_t       ratio  = 0;
    int64_t       base   = 0;
    dsv4_idx_mode mode   = dsv4_idx_mode::POS_MOD;
    // Position offset applied BEFORE the mode formula: the effective position
    // is last_pos + off. Lets the unrolled nt=2 verify pair drive per-token
    // scalar inputs (off = -1 for the pair's first token, 0 for the second).
    int64_t       off    = 0;
};

// Feature flag (env DSV4_CONSTANT_SHAPE): build the decode compressed-attention
// path with a depth-INVARIANT shape+topology so the CUDA graph REPLAYS across
// compression boundaries at all depths (instead of recapturing every `ratio`
// tokens, which makes graphs-on slower than graphs-off past ~250 tokens). Off
// by default until validated.
static bool dsv4_constant_shape_enabled() {
    static const bool v = getenv("DSV4_CONSTANT_SHAPE") != nullptr;
    return v;
}

// Kill-switch for the top-k GATHER decode path (the beyond-top_k long-context
// regime). With DSV4_CONSTANT_SHAPE set and this unset, decode past
// n_comp_visible > top_k gathers the selected 512 compressed rows via
// get_rows so the attention width stays [n_raw + top_k] at any depth.
static bool dsv4_topk_gather_disabled() {
    static const bool v = getenv("DSV4_NO_TOPK_GATHER") != nullptr;
    return v;
}

// Indexer-score width bucket (rows) for the gather path: the score/argsort
// shapes change only every DSV4_CS_BUCKET*ratio tokens (=8192 at ratio 4), so
// the decode graph replays in between instead of recapturing every boundary.
static constexpr int64_t DSV4_CS_BUCKET = 2048;

// Sparse top-k FA for ratio-4 prompt chunks (env DSV4_SPARSE_FA=1): instead of
// masking the FULL compressed width (dense FA cost grows linearly with depth,
// the dominant quadratic term of long prefills), pass a per-token index list
// [all raw rows ++ n_raw + comp top-k] into the DSA top-k FA kernel - the
// kernel then iterates only those rows and reads the mask through the indices.
// DSV4_SPARSE_FA=2 builds the FULL index list (same attended set as the dense
// path) - a correctness A/B mode, no speedup.
static int dsv4_sparse_fa_mode() {
    static const int v = [] {
        const char * e = getenv("DSV4_SPARSE_FA");
        return e == nullptr ? 0 : atoi(e);
    }();
    return v;
}

// GPU-built masks for the multi-token (prompt-chunk) paths. The CPU input
// masks cost an O(width x n_tokens) -INF fill + PCIe upload per ubatch (the
// dominant CPU-side quadratic term of long prefills: at 200K ctx the indexer
// causal mask alone is ~400 MB per 2048-token ubatch). Built on GPU from
// inp_pos with plain ggml ops instead. env DSV4_NO_GPU_MASKS reverts.
static bool dsv4_gpu_masks_disabled() {
    static const bool v = getenv("DSV4_NO_GPU_MASKS") != nullptr;
    return v;
}

// (dsv4_gpu_mask_comp_causal is defined below, after the scalar helpers.)

struct dsv4_mtp_module;
static dsv4_mtp_module & dsv4_mtp_get();
static int64_t dsv4_mtp_ring_pos_at(int64_t r);
static const float * dsv4_mtp_seed_data(size_t need);

class dsv4_graph_inputs : public llm_graph_input_i {
public:
    ggml_tensor * add_mask(
            ggml_context  * ctx,
            dsv4_mask_kind kind,
            int64_t        n0,
            int64_t        n1,
            int64_t        n_raw,
            int64_t        n_comp,
            int64_t        window,
            int64_t        ratio,
            const char   * name) {
        // Dedup by full parameter tuple: set_input fills the content purely from
        // (kind, ratio, window, shape, ubatch), so equal params => equal content.
        // Every layer sharing one tensor removes ~40 duplicate CPU mask fills +
        // uploads per token and the extra graph splits they caused.
        for (const auto & me : masks) {
            if (me.kind == kind && me.tensor->ne[0] == n0 && me.tensor->ne[1] == n1 &&
                me.n_raw == n_raw && me.n_comp == n_comp && me.window == window &&
                me.ratio == ratio) {
                return me.tensor;
            }
        }
        ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, n1, 1, 1);
        ggml_set_input(t);
        ggml_set_name(t, name);
        masks.push_back({ t, kind, n_raw, n_comp, window, ratio });
        return t;
    }

    // Record whether the graph was BUILT with the compressed-attention branch
    // for a given ratio (n_comp_visible > 0 at build time). Without this,
    // a decode graph built at pos < ratio (no comp branch, no comp masks) is
    // reused forever: nothing ever fails can_reuse, so layers with that ratio
    // never start attending their compressed cache (silent long-context loss
    // + deceptively fast decode).
    void note_comp_presence(int64_t ratio, bool has_comp) {
        for (const auto & e : presence) {
            if (e.ratio == ratio) {
                return;
            }
        }
        presence.push_back({ ratio, has_comp });
    }

    // Multi-token (prompt-chunk) graphs with GPU-built masks have no input
    // mask left to fail can_reuse when n_comp_visible grows -> record the
    // built width and force a rebuild when the expected width changes.
    void note_comp_width(int64_t ratio, int64_t width) {
        for (const auto & e : widths) {
            if (e.ratio == ratio) {
                return;
            }
        }
        widths.push_back({ ratio, width });
    }

    // Create a 1-element scalar INPUT whose value is recomputed each token from
    // (last_pos, ratio) per `mode` (see dsv4_idx_mode). Shape is constant [1] so
    // the decode graph stays reusable (CUDA graph replay); only DATA refreshes.
    // Dedup by (ratio, base, mode): every layer with the same params shares ONE
    // input, else we'd blow past GGML_SCHED_MAX_SPLIT_INPUTS.
    ggml_tensor * add_scalar_input(ggml_context * ctx, int64_t ratio, int64_t base,
                                   dsv4_idx_mode mode, ggml_type type, const char * name,
                                   int64_t off = 0) {
        for (const auto & ie : indices) {
            if (ie.ratio == ratio && ie.base == base && ie.mode == mode && ie.off == off) {
                return ie.tensor;
            }
        }
        ggml_tensor * t = ggml_new_tensor_1d(ctx, type, 1);
        ggml_set_input(t);
        ggml_set_name(t, name);
        indices.push_back({ t, ratio, base, mode, off });
        return t;
    }

    ggml_tensor * add_index(ggml_context * ctx, int64_t ratio, int64_t base, const char * name,
                            int64_t off = 0) {
        return add_scalar_input(ctx, ratio, base, dsv4_idx_mode::POS_MOD, GGML_TYPE_I32, name, off);
    }
    // RoPE position of the compressed chunk emitted this step (= last_pos+1-ratio).
    ggml_tensor * add_comp_pos_index(ggml_context * ctx, int64_t ratio, const char * name,
                                     int64_t off = 0) {
        return add_scalar_input(ctx, ratio, 0, dsv4_idx_mode::COMP_POS, GGML_TYPE_I32, name, off);
    }
    // Cache write row: real completed-chunk row on a compression boundary, else a
    // throwaway scratch row (so the always-built store never corrupts a real row).
    ggml_tensor * add_comp_row_index(ggml_context * ctx, int64_t ratio, int64_t scratch_row, const char * name,
                                     int64_t off = 0) {
        return add_scalar_input(ctx, ratio, scratch_row, dsv4_idx_mode::COMP_ROW, GGML_TYPE_I32, name, off);
    }
    // 1.0 on a compression boundary, else 0.0 (selects shifted vs unshifted state).
    ggml_tensor * add_boundary_flag(ggml_context * ctx, int64_t ratio, const char * name,
                                    int64_t off = 0) {
        return add_scalar_input(ctx, ratio, 0, dsv4_idx_mode::BOUNDARY_FLAG, GGML_TYPE_F32, name, off);
    }

    // --- MTP speculative-decode inputs (phase B) ------------------------
    // seed: HC state of the token preceding the batch (client-owned CPU copy).
    ggml_tensor * add_mtp_seed(ggml_context * ctx, int64_t n_embd, int64_t n_hc) {
        if (mtp_seed == nullptr) {
            mtp_seed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_hc);
            ggml_set_input(mtp_seed);
            ggml_set_name(mtp_seed, "mtp_seed_in");
        }
        return mtp_seed;
    }
    // ring write rows: pos % ring per batch token.
    ggml_tensor * add_mtp_ring_rows(ggml_context * ctx, int64_t n_tokens, int64_t ring) {
        if (mtp_ring_rows == nullptr) {
            mtp_ring_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
            ggml_set_input(mtp_ring_rows);
            ggml_set_name(mtp_ring_rows, "mtp_ring_rows");
            mtp_ring = ring;
        }
        return mtp_ring_rows;
    }
    // RoPE positions of the draft candidates: pos[nt-n_cand+j] + 1.
    ggml_tensor * add_mtp_cand_pos(ggml_context * ctx, int64_t n_cand) {
        if (mtp_cand_pos == nullptr) {
            mtp_cand_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cand);
            ggml_set_input(mtp_cand_pos);
            ggml_set_name(mtp_cand_pos, "mtp_cand_pos");
            mtp_n_cand = n_cand;
        }
        return mtp_cand_pos;
    }
    ggml_tensor * add_mtp_cand_pos2(ggml_context * ctx) {
        if (mtp_cand_pos2 == nullptr) {
            mtp_cand_pos2 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            ggml_set_input(mtp_cand_pos2);
            ggml_set_name(mtp_cand_pos2, "mtp_cand_pos2");
        }
        return mtp_cand_pos2;
    }
    // Level-2 candidate mask: single query at pos[nt-1]+2 over
    // [ring | batch-local | own-l1(last cand) | own-l2].
    ggml_tensor * add_mtp_mask2(ggml_context * ctx, int64_t ring, int64_t n_tokens,
                                int64_t window) {
        if (mtp_mask2 == nullptr) {
            mtp_mask2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ring + n_tokens + 2, 1);
            ggml_set_input(mtp_mask2);
            ggml_set_name(mtp_mask2, "mtp_mask2");
            mtp_window = window;
            mtp_ring   = ring;
        }
        return mtp_mask2;
    }
    // Candidate attention mask over [ring | batch-local KV | own KV] x n_cand.
    // Strict "<" against ring/local positions: a rejected draft's KV shares its
    // position with the reject-candidate's query and must stay invisible.
    ggml_tensor * add_mtp_mask(ggml_context * ctx, int64_t ring, int64_t n_tokens,
                               int64_t n_cand, int64_t window) {
        if (mtp_mask == nullptr) {
            mtp_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ring + n_tokens + n_cand, n_cand);
            ggml_set_input(mtp_mask);
            ggml_set_name(mtp_mask, "mtp_mask");
            mtp_window = window;
            mtp_ring   = ring;
        }
        return mtp_mask;
    }

    void set_input(const llama_ubatch * ubatch) override {
        const llama_pos last_pos = (ubatch && ubatch->pos && ubatch->n_tokens > 0)
            ? ubatch->pos[ubatch->n_tokens - 1] : 0;
        for (const auto & ie : indices) {
            if (ie.tensor == nullptr || ie.tensor->buffer == nullptr) {
                continue;
            }
            const int64_t r = ie.ratio;
            const llama_pos eff_pos = last_pos + ie.off;   // per-token offset (nt=2 unroll)
            const bool boundary = (r > 0) && (((eff_pos + 1) % r) == 0);
            switch (ie.mode) {
                case dsv4_idx_mode::POS_MOD: {
                    const int32_t v = (int32_t)(ie.base + (r > 0 ? (eff_pos % r) : 0));
                    ggml_backend_tensor_set(ie.tensor, &v, 0, sizeof(int32_t));
                } break;
                case dsv4_idx_mode::COMP_POS: {
                    const int32_t v = (int32_t)(eff_pos + 1 - r);
                    ggml_backend_tensor_set(ie.tensor, &v, 0, sizeof(int32_t));
                } break;
                case dsv4_idx_mode::COMP_ROW: {
                    const int32_t v = boundary ? (int32_t)(((eff_pos + 1) / r) - 1) : (int32_t)ie.base;
                    ggml_backend_tensor_set(ie.tensor, &v, 0, sizeof(int32_t));
                } break;
                case dsv4_idx_mode::BOUNDARY_FLAG: {
                    const float v = boundary ? 1.0f : 0.0f;
                    ggml_backend_tensor_set(ie.tensor, &v, 0, sizeof(float));
                } break;
            }
        }
        for (const auto & mask : masks) {
            GGML_ASSERT(mask.tensor != nullptr);
            if (mask.tensor->buffer == nullptr) {
                continue;
            }

            const int64_t n0 = mask.tensor->ne[0];
            const int64_t n1 = mask.tensor->ne[1];

            std::vector<float> data(n0*n1, -INFINITY);

            switch (mask.kind) {
                case dsv4_mask_kind::RAW_WINDOW:
                    fill_raw_window(data, n0, n1, mask.window, ubatch);
                    break;
                case dsv4_mask_kind::COMPRESS_CAUSAL:
                    fill_compress_causal(data, n0, n1, mask.ratio, 0, ubatch);
                    break;
                case dsv4_mask_kind::COMPRESS_FIXED:
                    // Fixed width: causal 0 for [0, n_comp_visible), the rest of
                    // [0, NC_FIXED) stays -INF (the data() init). Same fill logic.
                    fill_compress_causal(data, n0, n1, mask.ratio, 0, ubatch);
                    break;
                case dsv4_mask_kind::COMPRESS_BUCKET:
                    // Bucketed width: causal 0 for [0, n_comp_visible), the rest
                    // of the bucket stays -INF (the data() init). Same fill logic.
                    fill_compress_causal(data, n0, n1, mask.ratio, 0, ubatch);
                    break;
                case dsv4_mask_kind::ATTN_STATIC:
                    fill_raw_window(data, n0, n1, mask.window, ubatch);
                    fill_compress_causal(data, n0, n1, mask.ratio, mask.n_raw, ubatch);
                    break;
            }

            ggml_backend_tensor_set(mask.tensor, data.data(), 0, data.size()*sizeof(float));
        }

        // --- MTP speculative inputs (phase B) ---
        if (mtp_seed != nullptr && mtp_seed->buffer != nullptr) {
            const size_t n = (size_t) mtp_seed->ne[0] * mtp_seed->ne[1];
            ggml_backend_tensor_set(mtp_seed, dsv4_mtp_seed_data(n), 0, n * sizeof(float));
        }
        if (mtp_ring_rows != nullptr && mtp_ring_rows->buffer != nullptr && ubatch != nullptr) {
            const int64_t nt = mtp_ring_rows->ne[0];
            std::vector<int32_t> rows((size_t) nt, 0);
            for (int64_t i = 0; i < nt; ++i) {
                const llama_pos p = (ubatch->pos && i < (int64_t) ubatch->n_tokens)
                    ? ubatch->pos[i] : (llama_pos) i;
                rows[i] = (int32_t)(p % mtp_ring);
            }
            ggml_backend_tensor_set(mtp_ring_rows, rows.data(), 0, rows.size()*sizeof(int32_t));
        }
        if (mtp_cand_pos != nullptr && mtp_cand_pos->buffer != nullptr && ubatch != nullptr) {
            const int64_t nt = (int64_t) ubatch->n_tokens;
            std::vector<int32_t> cp((size_t) mtp_n_cand, 0);
            for (int64_t c = 0; c < mtp_n_cand; ++c) {
                const int64_t j = std::max<int64_t>(0, nt - mtp_n_cand + c);
                const llama_pos p = ubatch->pos ? ubatch->pos[j] : (llama_pos) j;
                cp[c] = (int32_t)(p + 1);
            }
            ggml_backend_tensor_set(mtp_cand_pos, cp.data(), 0, cp.size()*sizeof(int32_t));
        }
        if (mtp_cand_pos2 != nullptr && mtp_cand_pos2->buffer != nullptr && ubatch != nullptr) {
            const int64_t nt = (int64_t) ubatch->n_tokens;
            const int32_t v = (int32_t)((ubatch->pos ? ubatch->pos[nt - 1] : (llama_pos)(nt - 1)) + 2);
            ggml_backend_tensor_set(mtp_cand_pos2, &v, 0, sizeof(int32_t));
        }
        if (mtp_mask != nullptr && mtp_mask->buffer != nullptr && ubatch != nullptr) {
            const int64_t nt   = (int64_t) ubatch->n_tokens;
            const int64_t ring = mtp_ring;
            const int64_t n0   = mtp_mask->ne[0];   // ring + nt + n_cand
            const int64_t nc   = mtp_mask->ne[1];
            std::vector<float> data((size_t)(n0*nc), -INFINITY);
            for (int64_t c = 0; c < nc; ++c) {
                const int64_t j = std::max<int64_t>(0, nt - nc + c);
                const llama_pos q = (ubatch->pos ? ubatch->pos[j] : (llama_pos) j) + 1;
                // ring rows: strict '<' (a rejected draft's stale KV shares q's
                // position and must stay invisible), window-limited, empty = -inf
                for (int64_t r = 0; r < ring; ++r) {
                    const int64_t rp = dsv4_mtp_ring_pos_at(r);
                    if (rp >= 0 && rp < q && q - rp < mtp_window) {
                        data[(size_t)(c*n0 + r)] = 0.0f;
                    }
                }
                // batch-local KV: strict '<'
                for (int64_t i = 0; i < nt; ++i) {
                    const llama_pos p = ubatch->pos ? ubatch->pos[i] : (llama_pos) i;
                    if (p < q && q - p < mtp_window) {
                        data[(size_t)(c*n0 + ring + i)] = 0.0f;
                    }
                }
                // own KV: diagonal only
                data[(size_t)(c*n0 + ring + nt + c)] = 0.0f;
            }
            ggml_backend_tensor_set(mtp_mask, data.data(), 0, data.size()*sizeof(float));
        }
        if (mtp_mask2 != nullptr && mtp_mask2->buffer != nullptr && ubatch != nullptr) {
            const int64_t nt   = (int64_t) ubatch->n_tokens;
            const int64_t ring = mtp_ring;
            const int64_t n0   = mtp_mask2->ne[0];   // ring + nt + 2
            std::vector<float> data((size_t) n0, -INFINITY);
            const llama_pos q = (ubatch->pos ? ubatch->pos[nt - 1] : (llama_pos)(nt - 1)) + 2;
            for (int64_t r = 0; r < ring; ++r) {
                const int64_t rp = dsv4_mtp_ring_pos_at(r);
                if (rp >= 0 && rp < q && q - rp < mtp_window) data[(size_t) r] = 0.0f;
            }
            for (int64_t i = 0; i < nt; ++i) {
                const llama_pos p = ubatch->pos ? ubatch->pos[i] : (llama_pos) i;
                if (p < q && q - p < mtp_window) data[(size_t)(ring + i)] = 0.0f;
            }
            data[(size_t)(ring + nt)]     = 0.0f;   // own-l1 (draft1 @ q-1)
            data[(size_t)(ring + nt + 1)] = 0.0f;   // own-l2
            ggml_backend_tensor_set(mtp_mask2, data.data(), 0, data.size()*sizeof(float));
        }
    }

    // Graph reuse: the mask tensors have FIXED shapes baked at build time and
    // set_input() only refills their DATA. So the graph can be reused as long as
    // every mask would be built with the exact same shape for the new ubatch.
    // The only per-token variable is n_comp_visible = (last_pos+1)/ratio, which
    // changes only at compression boundaries (every `ratio` tokens). When it is
    // unchanged, the recurrent DSA state head is also unchanged (they advance
    // together), so the whole decode graph is identical and CUDA graphs replay.
    // When a boundary is crossed, return false -> the graph is rebuilt (rare).
    bool can_reuse(const llm_graph_params & params) override {
        const auto & ub = params.ubatch;
        const int64_t nt = ub.n_tokens;
        const llama_pos last_pos = ub.pos ? ub.pos[nt - 1] : (llama_pos)(nt - 1);
        // MTP spec inputs: shapes depend on nt and n_cand=min(nt,2).
        if (mtp_ring_rows != nullptr && mtp_ring_rows->ne[0] != nt) {
            return false;
        }
        if (mtp_mask != nullptr &&
            (mtp_mask->ne[1] != std::min<int64_t>(nt, 3) ||
             mtp_mask->ne[0] != mtp_ring + nt + mtp_mask->ne[1])) {
            return false;
        }
        if (mtp_mask2 != nullptr && mtp_mask2->ne[0] != mtp_ring + nt + 2) {
            return false;
        }
        // The comp branch must appear the moment n_comp_visible crosses 0 for
        // any ratio -> force a rebuild (see note_comp_presence).
        for (const auto & e : presence) {
            const bool now = e.ratio > 0 && ((last_pos + 1) / e.ratio) > 0;
            if (now != e.has_comp) {
                return false;
            }
        }
        // Chunk graphs with GPU-built masks: rebuild when the visible width
        // changes (see note_comp_width).
        for (const auto & e : widths) {
            if (e.ratio > 0 && (last_pos + 1) / e.ratio != e.width) {
                return false;
            }
        }
        for (const auto & m : masks) {
            if (m.tensor == nullptr || m.tensor->ne[1] != nt) {
                return false;
            }
            const int64_t ncv = m.ratio > 0 ? (last_pos + 1) / m.ratio : 0;
            int64_t want_n0;
            switch (m.kind) {
                case dsv4_mask_kind::RAW_WINDOW:      want_n0 = nt;       break;
                case dsv4_mask_kind::COMPRESS_CAUSAL: want_n0 = ncv;      break;
                case dsv4_mask_kind::ATTN_STATIC:     want_n0 = nt + ncv; break;
                // Fixed-width mask: ne[0] is a depth-invariant constant, so it
                // matches itself while n_comp_visible still fits the fixed width.
                // Once ncv outgrows it the build would switch to the top-k path,
                // so force a rebuild instead of silently reusing a stale view
                // that misses the newest compressed rows.
                case dsv4_mask_kind::COMPRESS_FIXED:
                    want_n0 = ncv <= m.tensor->ne[0] ? m.tensor->ne[0] : -1;
                    break;
                // Bucketed mask (top-k gather): recompute the bucketed width for
                // the new ubatch; reuse holds within a bucket, fails at bucket
                // crossings (every bucket*ratio tokens) -> rare rebuilds.
                case dsv4_mask_kind::COMPRESS_BUCKET:
                    want_n0 = std::min<int64_t>(GGML_PAD(ncv, m.window), m.n_comp);
                    break;
                default:                              return false;
            }
            if (m.tensor->ne[0] != want_n0) {
                return false;
            }
        }
        return true;
    }

private:
    static void fill_raw_window(
            std::vector<float> & data,
            int64_t              n0,
            int64_t              n1,
            int64_t              window,
            const llama_ubatch * ubatch) {
        GGML_ASSERT((int64_t) ubatch->n_tokens == n1);

        for (int64_t iq = 0; iq < n1; ++iq) {
            const llama_pos p1 = ubatch->pos ? ubatch->pos[iq] : (llama_pos) iq;

            for (int64_t ik = 0; ik < std::min<int64_t>(n0, ubatch->n_tokens); ++ik) {
                const llama_pos p0 = ubatch->pos ? ubatch->pos[ik] : (llama_pos) ik;

                if (p0 > p1) {
                    continue;
                }

                if (window > 0 && p1 - p0 >= window) {
                    continue;
                }

                data[iq*n0 + ik] = 0.0f;
            }
        }
    }

    static void fill_compress_causal(
            std::vector<float> & data,
            int64_t              n0,
            int64_t              n1,
            int64_t              ratio,
            int64_t              offset,
            const llama_ubatch * ubatch) {
        GGML_ASSERT(ratio > 0);

        const int64_t n_comp = n0 - offset;
        for (int64_t iq = 0; iq < n1; ++iq) {
            const llama_pos p1 = ubatch->pos ? ubatch->pos[iq] : (llama_pos) iq;
            const int64_t n_visible = (p1 + 1) / ratio;

            for (int64_t ic = 0; ic < std::min<int64_t>(n_comp, n_visible); ++ic) {
                data[iq*n0 + offset + ic] = 0.0f;
            }
        }
    }

    struct dsv4_presence_entry {
        int64_t ratio;
        bool    has_comp;
    };

    struct dsv4_width_entry {
        int64_t ratio;
        int64_t width;
    };

    std::vector<dsv4_mask_entry> masks;
    std::vector<dsv4_index_entry> indices;
    std::vector<dsv4_presence_entry> presence;
    std::vector<dsv4_width_entry> widths;

    // MTP speculative-decode inputs (phase B)
    ggml_tensor * mtp_seed      = nullptr;
    ggml_tensor * mtp_ring_rows = nullptr;
    ggml_tensor * mtp_cand_pos  = nullptr;
    ggml_tensor * mtp_mask      = nullptr;
    ggml_tensor * mtp_mask2     = nullptr;
    ggml_tensor * mtp_cand_pos2 = nullptr;
    int64_t mtp_ring   = 0;
    int64_t mtp_n_cand = 0;
    int64_t mtp_window = 0;
};

struct dsv4_rope_cfg {
    int32_t n_ctx_orig;
    float   freq_base;
    float   freq_scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
};

static ggml_tensor * dsv4_view_scale(ggml_context * ctx, ggml_tensor * scale, int64_t idx) {
    return ggml_view_2d(ctx, scale, 1, 1, scale->nb[0], idx * scale->nb[0]);
}

static ggml_tensor * dsv4_add_scalar(ggml_context * ctx, ggml_tensor * x, float value) {
    ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale_bias(ctx, x, 1.0f, value);
    return ggml_reshape(ctx, x, shape);
}

static ggml_tensor * dsv4_mul_scalar(ggml_context * ctx, ggml_tensor * x, float value) {
    ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale(ctx, x, value);
    return ggml_reshape(ctx, x, shape);
}

static ggml_tensor * dsv4_arange_i32(ggml_context * ctx, int64_t begin, int64_t end) {
    ggml_tensor * t = ggml_arange(ctx, (float) begin, (float) end, 1.0f);
    return ggml_cast(ctx, t, GGML_TYPE_I32);
}

static ggml_tensor * dsv4_new_filled_2d(ggml_context * ctx, int64_t n0, int64_t n1, float value) {
    return ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1), value);
}

static ggml_tensor * dsv4_new_filled_3d(ggml_context * ctx, int64_t n0, int64_t n1, int64_t n2, float value) {
    return ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2), value);
}

// Pad the flash-attention K/V width to a multiple of FATTN_KQ_STRIDE (256).
// The CUDA FA kernels for head size 512 REQUIRE K->ne[1] % 256 == 0
// (gqa_opt_applies in fattn.cu), otherwise the op silently falls back to the
// CPU backend: every ratio-4 prompt-chunk attention (width n_tokens +
// n_comp_visible, arbitrary) ran on the CPU - D2H copies of K/V/mask, an
// ~11 GB pinned host compute buffer re-allocated as the width grew
// (cudaMallocHost calls of up to 10 s per chunk), GPUs ~95% idle at depth.
// K/V rows are padded with ZEROS (ggml_pad), the F32 mask with -INF, so the
// padded tail carries zero attention weight - bit-identical output.
static void dsv4_pad_fattn_width(ggml_context * ctx, ggml_tensor ** k_all, ggml_tensor ** attn_mask) {
    const int64_t w   = (*k_all)->ne[2];
    const int64_t pad = ((w + 255) / 256) * 256 - w;
    if (pad == 0) {
        return;
    }
    // NB: CUDA ggml_pad asserts F32, so pad K via an F16 zero-fill + concat
    // (fill.cu supports F16).
    ggml_tensor * kz = ggml_fill(ctx,
            ggml_new_tensor_3d(ctx, (*k_all)->type, (*k_all)->ne[0], 1, pad), 0.0f);
    *k_all = ggml_concat(ctx, *k_all, kz, 2);
    ggml_tensor * mz = dsv4_new_filled_2d(ctx, pad, (*attn_mask)->ne[1], -INFINITY);
    *attn_mask = ggml_concat(ctx, *attn_mask, mz, 0);
}

// mask[i, q] = i < (pos_q + 1)/ratio ? 0 : -1e9   (compressed causal), on GPU.
// i < (p+1)/r  <=>  (i+1)*r <= p+1  <=>  p + 1.5 - (i+1)*r > 0 for integers.
static ggml_tensor * dsv4_gpu_mask_comp_causal(
        ggml_context * ctx,
        ggml_tensor  * pos,      // i32 [n_tokens]
        int64_t        n_comp,
        int64_t        ratio,
        int64_t        n_tokens) {
    ggml_tensor * ar = ggml_arange(ctx, 1.0f, (float) n_comp + 0.5f, 1.0f);   // [n_comp] = i+1
    ar = dsv4_mul_scalar(ctx, ar, -(float) ratio);                            // -(i+1)*ratio
    ar = ggml_repeat_4d(ctx, ggml_reshape_2d(ctx, ar, n_comp, 1), n_comp, n_tokens, 1, 1);
    ggml_tensor * pf = ggml_cast(ctx, pos, GGML_TYPE_F32);                    // [n_tokens]
    pf = dsv4_add_scalar(ctx, pf, 1.5f);                                      // p + 1.5
    pf = ggml_reshape_2d(ctx, pf, 1, n_tokens);
    ggml_tensor * vis = ggml_step(ctx, ggml_add(ctx, ar, pf));                // 1 if visible
    return dsv4_mul_scalar(ctx, dsv4_add_scalar(ctx, vis, -1.0f), 1.0e9f);    // 0 / -1e9
}

static dsv4_state_layout dsv4_make_state_layout(int64_t compress_ratio, int64_t head_dim) {
    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t width = coff * head_dim;
    const int64_t rows  = coff * compress_ratio;
    return { width, rows, width * rows };
}

static ggml_tensor * dsv4_view_cols(
        ggml_context * ctx,
        ggml_tensor  * x,
        int64_t        n0,
        int64_t        n1,
        int64_t        off0,
        int64_t        off1) {
    return ggml_view_2d(ctx, x, n0, n1, x->nb[1], off1*x->nb[1] + off0*x->nb[0]);
}

static ggml_tensor * dsv4_view_state_segment(
        ggml_context * ctx,
        ggml_tensor  * state,
        int64_t        offset,
        int64_t        width,
        int64_t        rows) {
    return ggml_view_2d(ctx, state, width, rows, width*state->nb[0], offset*state->nb[0]);
}

static void dsv4_store_state_segment(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        ggml_tensor  * src,
        ggml_tensor  * dst,
        int64_t        state_size,
        int64_t        head,
        int64_t        offset) {
    const int64_t n = ggml_nelements(src);
    src = ggml_cont(ctx, src);
    src = ggml_reshape_1d(ctx, src, n);

    ggml_tensor * view = ggml_view_1d(ctx, dst, n, (head*state_size + offset)*ggml_element_size(dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, view));
}

// Returns the POST-STORE cache tensor. Threading this returned tensor into the
// subsequent cache read (instead of a fresh get_dsv4_attn_k view) creates the
// RAW dependency edge store->read that ggml's scheduler needs: the store is a
// side-effect leaf (set_rows/cpy) with no tensor edge to the later read, so the
// read can otherwise be scheduled before/concurrent with the write. Submission
// order masks this in boundary-gated lossless (store at line ~1597 expanded
// before the read at ~1608), but in always-build/const-shape the store runs
// EVERY token via ggml_set_rows whose multi-GPU routing follows the SOURCE
// device, decoupling it from the read's device/stream -> race -> garbage. The
// explicit edge fixes both. (Diagnosed by codex panel, 2026-06-27.)
static ggml_tensor * dsv4_store_cache_rows(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        ggml_tensor  * cache,
        ggml_tensor  * src,
        int64_t        row_start,
        int64_t        n_rows,
        ggml_tensor  * row_idx = nullptr) {
    if (row_idx) {
        // Constant-shape decode: ALWAYS store exactly one row via ggml_set_rows to
        // a data-driven row (real completed-chunk row on a boundary, throwaway
        // scratch row otherwise). Keeps the store node ALWAYS present in the graph
        // (no DCE on non-boundary tokens) -> the decode graph topology is
        // depth-invariant so CUDA graphs REPLAY across compression boundaries.
        // set_rows is layer-split-safe (cache+src co-located on one GPU).
        ggml_tensor * src1 = ggml_reshape_2d(ctx, ggml_cont(ctx, src), cache->ne[0], 1);
        ggml_tensor * updated = ggml_set_rows(ctx, cache, src1, row_idx);
        ggml_build_forward_expand(gf, updated);
        return updated;
    }
    if (n_rows <= 0) {
        return cache;
    }

    src = ggml_cont(ctx, src);
    src = ggml_reshape_2d(ctx, src, cache->ne[0], n_rows);

    // Avoid ggml_set_rows here: on multi-GPU, sched routes set_rows by SOURCE
    // device, but the cache destination has its own device affinity → illegal
    // memory access when those differ. ggml_cpy into a contiguous view of
    // cache routes correctly by dst affinity (same pattern as
    // dsv4_store_state_segment, which works in production multi-GPU).
    ggml_tensor * cache_view = ggml_view_2d(ctx, cache,
            cache->ne[0], n_rows,
            cache->nb[1],
            row_start * cache->nb[1]);
    ggml_tensor * cpy = ggml_cpy(ctx, src, cache_view);
    ggml_build_forward_expand(gf, cpy);
    // Return a full-shape view of cache that depends on the cpy, so consumers
    // that read this returned tensor are ordered after the write.
    ggml_tensor * updated = ggml_view_tensor(ctx, cache);
    updated->src[0] = cpy;
    return updated;
}

static dsv4_rope_cfg dsv4_make_rope_cfg(
        const llama_hparams & hparams,
        const llama_cparams  & cparams,
        uint32_t              compress_ratio) {
    if (compress_ratio == 0) {
        return {
            0,
            hparams.rope_freq_base_train,
            1.0f,
            0.0f,
            1.0f,
            cparams.yarn_beta_fast,
            cparams.yarn_beta_slow,
        };
    }

    float attn_factor = 1.0f;
    if (cparams.yarn_ext_factor != 0.0f && cparams.rope_freq_scale > 0.0f) {
        // DeepSeek V4 uses YaRN-style frequency interpolation for compressed RoPE,
        // but the reference implementation does not apply YaRN's magnitude scale.
        attn_factor /= 1.0f + 0.1f * std::log(1.0f / cparams.rope_freq_scale);
    }

    return {
        (int32_t) cparams.n_ctx_orig_yarn,
        hparams.compress_rope_freq_base > 0.0f ? hparams.compress_rope_freq_base : cparams.rope_freq_base,
        cparams.rope_freq_scale,
        cparams.yarn_ext_factor,
        attn_factor,
        cparams.yarn_beta_fast,
        cparams.yarn_beta_slow,
    };
}

static ggml_tensor * dsv4_view_base(ggml_context * ctx, ggml_tensor * base, int64_t n, int64_t off) {
    return ggml_view_2d(ctx, base, n, 1, base->nb[0], off * base->nb[0]);
}

static ggml_tensor * dsv4_apply_rope_tail(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * inp_pos,
        int64_t        n_embd_head,
        int64_t        n_head,
        int64_t        n_tokens,
        int64_t        n_rot,
        int            rope_type,
        int32_t        n_ctx_orig,
        float          freq_base,
        float          freq_scale,
        float          ext_factor,
        float          attn_factor,
        float          beta_fast,
        float          beta_slow,
        bool           inverse) {
    GGML_ASSERT(x->ne[0] == n_embd_head);
    GGML_ASSERT(x->ne[1] == n_head);
    GGML_ASSERT(x->ne[2] == n_tokens);

    if (n_rot == n_embd_head) {
        return inverse
            ? ggml_rope_ext_back(ctx, x, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
            : ggml_rope_ext     (ctx, x, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }

    const int64_t n_nope = n_embd_head - n_rot;
    GGML_ASSERT(n_nope > 0);

    return ggml_dsv4_rope_tail(ctx, x, inp_pos, nullptr, n_rot, rope_type,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, inverse);
}

static dsv4_hc_mix dsv4_hc_pre(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens,
        float          norm_eps,
        int            sinkhorn_iters,
        float          hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;
    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat); // [mix_hc, n_tokens]
    ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(ctx, mixes, hc_scale, hc_base, n_hc, sinkhorn_iters, hc_eps);
    ggml_tensor * pre = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], 0);
    ggml_tensor * post = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], n_hc * split->nb[0]);
    ggml_tensor * comb = ggml_view_2d(ctx, split, n_hc * n_hc, n_tokens, split->nb[1], 2 * n_hc * split->nb[0]);
    if (n_tokens != 1) {
        pre = ggml_cont(ctx, pre);
        post = ggml_cont(ctx, post);
        comb = ggml_cont(ctx, comb);
    }
    comb = ggml_reshape_3d(ctx, comb, n_hc, n_hc, n_tokens); // [src_hc, dst_hc, n_tokens]
    ggml_tensor * y = ggml_dsv4_hc_weighted_sum(ctx, x, pre);
    return { y, mixes, pre, post, comb };
}

static ggml_tensor * dsv4_hc_post(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * residual,
        ggml_tensor  * post,
        ggml_tensor  * comb,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens) {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == n_tokens);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == n_hc);
    GGML_ASSERT(residual->ne[2] == n_tokens);
    GGML_ASSERT(post->ne[0] == n_hc);
    GGML_ASSERT(post->ne[1] == n_tokens);
    GGML_ASSERT(comb->ne[0] == n_hc);
    GGML_ASSERT(comb->ne[1] == n_hc);
    GGML_ASSERT(comb->ne[2] == n_tokens);

    return ggml_dsv4_hc_expand(ctx, x, residual, post, comb);
}

static ggml_tensor * dsv4_hc_head(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens,
        float          norm_eps,
        float          hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;

    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);

    ggml_tensor * pre = ggml_mul_mat(ctx, hc_fn, flat); // [hc, n_tokens]
    pre = ggml_mul(ctx, pre, dsv4_view_scale(ctx, hc_scale, 0));
    pre = ggml_add(ctx, pre, dsv4_view_base(ctx, hc_base, n_hc, 0));
    pre = dsv4_add_scalar(ctx, ggml_sigmoid(ctx, pre), hc_eps);

    return ggml_dsv4_hc_weighted_sum(ctx, x, pre);
}

static ggml_tensor * dsv4_grouped_out(
        ggml_context * ctx,
        ggml_tensor  * o,
        ggml_tensor  * wo_a,
        ggml_tensor  * wo_b,
        int64_t        n_embd_head,
        int64_t        n_head,
        int64_t        n_groups,
        int64_t        o_lora_rank,
        int64_t        n_tokens) {
    GGML_ASSERT(n_head % n_groups == 0);

    const int64_t group_heads = n_head / n_groups;
    const int64_t group_dim   = n_embd_head * group_heads;

    o = ggml_cont(ctx, o);
    o = ggml_reshape_3d(ctx, o, group_dim, n_groups, n_tokens);

    ggml_tensor * wo_a_g = ggml_reshape_3d(ctx, wo_a, group_dim, o_lora_rank, n_groups);
    ggml_tensor * ids = ggml_arange(ctx, 0.0f, float(n_groups), 1.0f);
    ids = ggml_cast(ctx, ids, GGML_TYPE_I32);
    ids = ggml_repeat_4d(ctx, ids, n_groups, n_tokens, 1, 1);

    ggml_tensor * low = ggml_mul_mat_id(ctx, wo_a_g, o, ids); // [o_lora_rank, n_groups, n_tokens]
    low = ggml_reshape_2d(ctx, low, o_lora_rank * n_groups, n_tokens);

    return ggml_mul_mat(ctx, wo_b, low);
}

static ggml_tensor * dsv4_softmax_pool_ratio(
        ggml_context * ctx,
        ggml_tensor  * kv,
        ggml_tensor  * score) {
    score = ggml_soft_max(ctx, score);
    ggml_tensor * pooled = ggml_mul(ctx, kv, score);
    pooled = ggml_sum_rows(ctx, pooled);
    return ggml_reshape_2d(ctx, pooled, kv->ne[1], kv->ne[2]);
}

static ggml_tensor * dsv4_shift_overlap_state(
        ggml_context * ctx,
        ggml_tensor  * x,
        float          pad_value) {
    const int64_t n_embd  = x->ne[0];
    const int64_t ratio   = x->ne[1];
    const int64_t n_comp  = x->ne[2];

    ggml_tensor * first = ggml_view_3d(ctx, x, n_embd, ratio, 1,
            x->nb[1], x->nb[2], 0);
    ggml_tensor * pad = ggml_fill(ctx, ggml_cont(ctx, first), pad_value);

    if (n_comp == 1) {
        return pad;
    }

    ggml_tensor * prev = ggml_view_3d(ctx, x, n_embd, ratio, n_comp - 1,
            x->nb[1], x->nb[2], 0);
    return ggml_concat(ctx, pad, prev, 2);
}

static ggml_tensor * dsv4_build_compressor_prefill(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        ggml_tensor        * pos,
        int64_t              n_embd_head,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    GGML_ASSERT(compress_ratio > 0);
    const int64_t n_comp = n_tokens / compress_ratio;
    GGML_ASSERT(n_comp > 0);

    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t n_kv = coff * n_embd_head;
    const int64_t cutoff = n_comp * compress_ratio;

    ggml_tensor * kv = ggml_mul_mat(ctx, wkv, x);       // [coff*head_dim, n_tokens]
    ggml_tensor * score = ggml_mul_mat(ctx, wgate, x);  // [coff*head_dim, n_tokens]

    kv = ggml_view_3d(ctx, kv, n_kv, compress_ratio, n_comp,
            kv->nb[1],
            kv->nb[1] * compress_ratio,
            0);
    score = ggml_view_3d(ctx, score, n_kv, compress_ratio, n_comp,
            score->nb[1],
            score->nb[1] * compress_ratio,
            0);
    GGML_ASSERT(cutoff <= n_tokens);

    ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    score = ggml_add(ctx, score, ggml_repeat(ctx, ape_f, score));

    if (coff == 1) {
        kv = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));       // [ratio, head_dim, n_comp]
        score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3)); // [ratio, head_dim, n_comp]
        kv = dsv4_softmax_pool_ratio(ctx, kv, score);                // [head_dim, n_comp]
    } else {
        ggml_tensor * kv_prev = ggml_view_3d(ctx, kv, n_embd_head, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], 0);
        ggml_tensor * kv_curr = ggml_view_3d(ctx, kv, n_embd_head, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], n_embd_head * kv->nb[0]);
        ggml_tensor * score_prev = ggml_view_3d(ctx, score, n_embd_head, compress_ratio, n_comp,
                score->nb[1], score->nb[2], 0);
        ggml_tensor * score_curr = ggml_view_3d(ctx, score, n_embd_head, compress_ratio, n_comp,
                score->nb[1], score->nb[2], n_embd_head * score->nb[0]);

        kv_prev    = dsv4_shift_overlap_state(ctx, kv_prev,    0.0f);
        score_prev = dsv4_shift_overlap_state(ctx, score_prev, -INFINITY);

        kv_prev    = ggml_cont(ctx, ggml_permute(ctx, kv_prev,    1, 0, 2, 3)); // [ratio, head_dim, n_comp]
        kv_curr    = ggml_cont(ctx, ggml_permute(ctx, kv_curr,    1, 0, 2, 3));
        score_prev = ggml_cont(ctx, ggml_permute(ctx, score_prev, 1, 0, 2, 3));
        score_curr = ggml_cont(ctx, ggml_permute(ctx, score_curr, 1, 0, 2, 3));

        kv    = ggml_concat(ctx, kv_prev,    kv_curr,    0); // [2*ratio, head_dim, n_comp]
        score = ggml_concat(ctx, score_prev, score_curr, 0);
        kv = dsv4_softmax_pool_ratio(ctx, kv, score);        // [head_dim, n_comp]
    }

    kv = ggml_rms_norm(ctx, kv, norm_eps);
    kv = ggml_mul(ctx, kv, norm);
    kv = ggml_reshape_3d(ctx, kv, n_embd_head, 1, n_comp);

    kv = dsv4_apply_rope_tail(ctx, kv, pos,
            n_embd_head, 1, n_comp, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

    return kv;
}

static dsv4_state_pair dsv4_build_compressor_prefill_state(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * wkv,
        ggml_tensor  * wgate,
        ggml_tensor  * ape,
        int64_t        head_dim,
        int64_t        n_tokens,
        int64_t        compress_ratio) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);

    const int64_t cutoff    = (n_tokens / compress_ratio) * compress_ratio;
    const int64_t remainder = n_tokens - cutoff;

    ggml_tensor * kv    = ggml_mul_mat(ctx, wkv,    x); // [width, n_tokens]
    ggml_tensor * score = ggml_mul_mat(ctx, wgate,  x);
    ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    if (compress_ratio == 4) {
        ggml_tensor * kv_prev    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
        ggml_tensor * score_prev = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

        if (cutoff >= compress_ratio) {
            kv_prev = ggml_view_2d(ctx, kv, layout.width, compress_ratio, kv->nb[1], (cutoff - compress_ratio)*kv->nb[1]);
            score_prev = ggml_view_2d(ctx, score, layout.width, compress_ratio, score->nb[1], (cutoff - compress_ratio)*score->nb[1]);
            score_prev = ggml_add(ctx, score_prev, ape_f);
        }

        ggml_tensor * kv_curr    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
        ggml_tensor * score_curr = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

        if (remainder > 0) {
            ggml_tensor * kv_rem = ggml_view_2d(ctx, kv, layout.width, remainder, kv->nb[1], cutoff*kv->nb[1]);
            ggml_tensor * sc_rem = ggml_view_2d(ctx, score, layout.width, remainder, score->nb[1], cutoff*score->nb[1]);
            sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0));

            if (remainder == compress_ratio) {
                kv_curr = kv_rem;
                score_curr = sc_rem;
            } else {
                kv_curr = ggml_concat(ctx, kv_rem,
                        dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, 0.0f), 1);
                score_curr = ggml_concat(ctx, sc_rem,
                        dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, -INFINITY), 1);
            }
        }

        return {
            ggml_concat(ctx, kv_prev,    kv_curr,    1),
            ggml_concat(ctx, score_prev, score_curr, 1),
        };
    }

    ggml_tensor * kv_state    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
    ggml_tensor * score_state = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

    if (remainder > 0) {
        ggml_tensor * kv_rem = ggml_view_2d(ctx, kv, layout.width, remainder, kv->nb[1], cutoff*kv->nb[1]);
        ggml_tensor * sc_rem = ggml_view_2d(ctx, score, layout.width, remainder, score->nb[1], cutoff*score->nb[1]);
        sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0));

        if (remainder == compress_ratio) {
            kv_state = kv_rem;
            score_state = sc_rem;
        } else {
            kv_state = ggml_concat(ctx, kv_rem,
                    dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, 0.0f), 1);
            score_state = ggml_concat(ctx, sc_rem,
                    dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, -INFINITY), 1);
        }
    }

    return { kv_state, score_state };
}

static ggml_tensor * dsv4_pool_decode_state(
        ggml_context * ctx,
        ggml_tensor  * kv,
        ggml_tensor  * score,
        ggml_tensor  * norm,
        ggml_tensor  * pos,
        int64_t        head_dim,
        int64_t        n_rot,
        int            rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float          norm_eps) {
    const int64_t n_rows = kv->ne[1];
    kv    = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, kv)),    n_rows, head_dim, 1);
    score = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, score)), n_rows, head_dim, 1);

    ggml_tensor * pooled = dsv4_softmax_pool_ratio(ctx, kv, score);
    pooled = ggml_rms_norm(ctx, pooled, norm_eps);
    pooled = ggml_mul(ctx, pooled, norm);
    pooled = ggml_reshape_3d(ctx, pooled, head_dim, 1, 1);

    return dsv4_apply_rope_tail(ctx, pooled, pos,
            head_dim, 1, 1, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
}

static dsv4_decode_compressor dsv4_build_compressor_decode_projected(
        ggml_context       * ctx,
        ggml_tensor        * kv_cur,
        ggml_tensor        * sc_cur,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps,
        ggml_tensor        * row_idx,
        ggml_tensor        * comp_pos_idx = nullptr,
        ggml_tensor        * boundary_flag = nullptr);

static dsv4_decode_compressor dsv4_build_compressor_decode(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps,
        ggml_tensor        * ape_idx = nullptr,
        ggml_tensor        * row_idx = nullptr,
        ggml_tensor        * comp_pos_idx = nullptr,
        ggml_tensor        * boundary_flag = nullptr) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
    const int64_t pos_mod = pos % compress_ratio;

    ggml_tensor * kv_cur = ggml_mul_mat(ctx, wkv, x);       // [width, 1]
    ggml_tensor * sc_cur = ggml_mul_mat(ctx, wgate, x);
    ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    // APE row select: use a refreshed index input (get_rows) so the graph can be
    // reused across tokens; fall back to the baked view offset when no index.
    ggml_tensor * ape_row = ape_idx
        ? ggml_get_rows(ctx, ape_f, ape_idx)
        : ggml_view_2d(ctx, ape_f, layout.width, 1, ape_f->nb[1], pos_mod*ape_f->nb[1]);
    sc_cur = ggml_add(ctx, sc_cur, ape_row);

    return dsv4_build_compressor_decode_projected(ctx,
            kv_cur, sc_cur,
            prev_kv_state, prev_score_state,
            norm,
            head_dim, n_rot, pos, compress_ratio,
            rope_type, rope_cfg, norm_eps, row_idx,
            comp_pos_idx, boundary_flag);
}

static dsv4_decode_compressor dsv4_build_compressor_decode_projected(
        ggml_context       * ctx,
        ggml_tensor        * kv_cur,
        ggml_tensor        * sc_cur,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps,
        ggml_tensor        * row_idx,
        ggml_tensor        * comp_pos_idx,
        ggml_tensor        * boundary_flag) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
    const int64_t pos_mod = pos % compress_ratio;
    const int64_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = (pos + 1) % compress_ratio == 0;
    // Constant-shape mode (boundary_flag supplied): ALWAYS build the compression
    // block so the decode graph topology is depth-invariant (CUDA-graph replay
    // across compression boundaries). The state-shift side effect is applied only
    // on real boundaries via an exact blend (flag in {0,1}); kv_comp is produced
    // every token but the caller routes its store to a scratch row off-boundary.
    const bool const_shape = (boundary_flag != nullptr);

    // Single-row write into the recurrent state ring buffer.
    //
    // When row_idx (a refreshed per-token index input) is supplied, use
    // ggml_set_rows so the write slot is data-driven, NOT baked into the graph
    // as a view offset -> the decode graph can be reused across tokens (CUDA
    // graph replay). In layer-split the state and the source live on the same
    // device, so the historical multi-GPU set_rows routing crash does not apply.
    //
    // Fallback (row_idx == nullptr, e.g. the multi-token chunk path which never
    // reuses): the original cpy-into-baked-view. It returns a FULL-shape view of
    // dst with a manual dependency on the cpy so consumers wait for the write.
    auto cpy_into_row = [&](ggml_tensor * dst, ggml_tensor * row_src) -> ggml_tensor * {
        if (row_idx) {
            return ggml_set_rows(ctx, dst, row_src, row_idx);
        }
        ggml_tensor * row_view = ggml_view_2d(ctx, dst,
                dst->ne[0], 1,
                dst->nb[1],
                row * dst->nb[1]);
        ggml_tensor * cpy = ggml_cpy(ctx, row_src, row_view);
        ggml_tensor * full_state = ggml_view_tensor(ctx, dst);
        full_state->src[0] = cpy;  // dependency: full_state's consumers wait for cpy
        return full_state;
    };
    ggml_tensor * kv_state    = cpy_into_row(prev_kv_state,    kv_cur);
    ggml_tensor * score_state = cpy_into_row(prev_score_state, sc_cur);
    ggml_tensor * kv_comp = nullptr;

    // Exact blend selector: flag in {0,1} -> a*flag + b*(1-flag) is bit-exact
    // (0.0*finite = 0, x+0 = x). Picks `a` on boundary (flag=1), `b` otherwise.
    auto blend = [&](ggml_tensor * a, ggml_tensor * b) -> ggml_tensor * {
        ggml_tensor * one_minus = ggml_scale_bias(ctx, boundary_flag, -1.0f, 1.0f); // 1 - flag
        return ggml_add(ctx, ggml_mul(ctx, a, boundary_flag), ggml_mul(ctx, b, one_minus));
    };

    // ALWAYS-BUILD (const_shape): run the compression block every token so the
    // decode graph topology is depth-invariant (CUDA-graph replay across
    // compression boundaries). The state-shift side effect is applied only on
    // real boundaries via the NaN-safe blend() below (flag in {0,1}); off-boundary
    // the emitted partial kv_comp is routed by the caller to a scratch cache row.
    // (This corrupted decode into "<<<<" until the blend's 0*(-INF)=NaN on the
    // score state's empty-slot sentinels was fixed by the clamp below.)
    // Boundary-gated (`should_compress`) remains the path when const_shape is off.
    if (should_compress || const_shape) {
        ggml_tensor * kv_pool;
        ggml_tensor * score_pool;

        if (compress_ratio == 4) {
            ggml_tensor * kv_prev = dsv4_view_cols(ctx, kv_state,    head_dim, compress_ratio, 0,        0);
            ggml_tensor * kv_curr = dsv4_view_cols(ctx, kv_state,    head_dim, compress_ratio, head_dim, compress_ratio);
            ggml_tensor * sc_prev = dsv4_view_cols(ctx, score_state, head_dim, compress_ratio, 0,        0);
            ggml_tensor * sc_curr = dsv4_view_cols(ctx, score_state, head_dim, compress_ratio, head_dim, compress_ratio);

            kv_pool    = ggml_concat(ctx, kv_prev, kv_curr, 1);
            score_pool = ggml_concat(ctx, sc_prev, sc_curr, 1);

            ggml_tensor * shifted_kv    = dsv4_view_cols(ctx, kv_state,    layout.width, compress_ratio, 0, compress_ratio);
            ggml_tensor * shifted_score = dsv4_view_cols(ctx, score_state, layout.width, compress_ratio, 0, compress_ratio);
            ggml_tensor * new_kv_state    = ggml_concat(ctx, shifted_kv,    shifted_kv,    1);
            ggml_tensor * new_score_state = ggml_concat(ctx, shifted_score, shifted_score, 1);
            // const_shape: apply the shift only on real boundaries (blend);
            // otherwise the original unconditional shift (this path only runs at a
            // boundary when !const_shape).
            // THE const-shape NaN fix: the SCORE state carries -INF empty-slot
            // sentinels (dsv4_new_filled_2d -INFINITY). blend's (1-flag)*score term
            // computes 0*(-INF) = NaN at a boundary (flag==1), which propagates
            // through dsv4_softmax_pool_ratio's soft_max -> the compressed KV becomes
            // NaN -> attention degenerates to "<<<<". Clamp the score operands to a
            // finite floor first: soft_max treats -3e4 identically to -INF (weight 0),
            // so this is numerically equivalent AND NaN-safe. The KV state is finite
            // (0-filled empties) so its blend needs no clamp. (Isolated via panel +
            // bisection 2026-06-27; the "0.0*finite=0" blend comment missed that the
            // score operand is not finite.)
            auto clamp_fin = [&](ggml_tensor * t) { return ggml_clamp(ctx, t, -3.0e4f, 3.0e4f); };
            kv_state    = const_shape ? blend(new_kv_state, kv_state) : new_kv_state;
            score_state = const_shape
                ? blend(clamp_fin(new_score_state), clamp_fin(score_state))
                : new_score_state;
        } else {
            kv_pool    = kv_state;
            score_pool = score_state;
        }

        ggml_tensor * comp_pos = const_shape
            ? comp_pos_idx
            : dsv4_arange_i32(ctx, pos + 1 - compress_ratio, pos + 2 - compress_ratio);
        kv_comp = dsv4_pool_decode_state(ctx, kv_pool, score_pool, norm, comp_pos,
                head_dim, n_rot, rope_type, rope_cfg, norm_eps);
    }

    return { kv_state, score_state, kv_comp };
}

static dsv4_decode_compressor dsv4_build_compressor_decode_chunk(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        const llama_ubatch & ubatch,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);

    ggml_tensor * kv_all = ggml_mul_mat(ctx, wkv,   x); // [width, n_tokens]
    ggml_tensor * sc_all = ggml_mul_mat(ctx, wgate, x);
    ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    ggml_tensor * kv_state    = prev_kv_state;
    ggml_tensor * score_state = prev_score_state;
    ggml_tensor * kv_comp     = nullptr;

    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_pos pos = ubatch.pos ? ubatch.pos[i] : (llama_pos) i;
        const int64_t pos_mod = pos % compress_ratio;

        ggml_tensor * kv_cur = ggml_view_2d(ctx, kv_all, layout.width, 1, kv_all->nb[1], i*kv_all->nb[1]);
        ggml_tensor * sc_cur = ggml_view_2d(ctx, sc_all, layout.width, 1, sc_all->nb[1], i*sc_all->nb[1]);
        sc_cur = ggml_add(ctx, sc_cur, ggml_view_2d(ctx, ape_f, layout.width, 1, ape_f->nb[1], pos_mod*ape_f->nb[1]));

        dsv4_decode_compressor dec = dsv4_build_compressor_decode_projected(ctx,
                kv_cur,
                sc_cur,
                kv_state,
                score_state,
                norm,
                head_dim,
                n_rot,
                pos,
                compress_ratio,
                rope_type,
                rope_cfg,
                norm_eps,
                /*row_idx=*/nullptr);

        kv_state    = dec.kv_state;
        score_state = dec.score_state;
        if (dec.kv_comp != nullptr) {
            kv_comp = kv_comp == nullptr ? dec.kv_comp : ggml_concat(ctx, kv_comp, dec.kv_comp, 2);
        }
    }

    return { kv_state, score_state, kv_comp };
}

// Kill-switch for the batched prompt-chunk compressor below.
static bool dsv4_batched_chunk_disabled() {
    static const bool v = getenv("DSV4_NO_BATCHED_CHUNK") != nullptr;
    return v;
}

// Batched replacement for dsv4_build_compressor_decode_chunk on ratio-ALIGNED
// prompt chunks (pos[0] % ratio == 0 && n_tokens % ratio == 0). The per-token
// loop builds a full compressor subgraph per token (~1000 graph objects and
// ~1000 kernel launches per token -> ~500K per 512-token ubatch; ggml arena
// overflow at ubatch >= 1024, and the launch spam keeps the GPUs ~90% idle
// during prefill). This pools every compression window of the ubatch at once,
// exactly like the pos-0 prefill path (dsv4_build_compressor_prefill), with
// two additions: the ratio-4 overlap window of chunk 0 is SEEDED from the
// carried recurrent state (rows [0, ratio) hold the previous ubatch's last
// full window), and the end state is emitted like
// dsv4_build_compressor_prefill_state does for the aligned case.
static dsv4_decode_compressor dsv4_build_compressor_chunk_batched(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        llama_pos            first_pos,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
    const int64_t n_comp = n_tokens / compress_ratio;
    const int64_t coff   = compress_ratio == 4 ? 2 : 1;
    const int64_t n_kv   = coff * head_dim;

    GGML_ASSERT(first_pos % compress_ratio == 0);
    GGML_ASSERT(n_comp > 0 && n_comp * compress_ratio == n_tokens);
    GGML_ASSERT(layout.width == n_kv);

    ggml_tensor * kv_all = ggml_mul_mat(ctx, wkv,   x); // [width, n_tokens]
    ggml_tensor * sc_all = ggml_mul_mat(ctx, wgate, x);
    ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    // ---- pooled compressed rows (mirrors dsv4_build_compressor_prefill) ----
    ggml_tensor * kv = ggml_view_3d(ctx, kv_all, n_kv, compress_ratio, n_comp,
            kv_all->nb[1], kv_all->nb[1] * compress_ratio, 0);
    ggml_tensor * score = ggml_view_3d(ctx, sc_all, n_kv, compress_ratio, n_comp,
            sc_all->nb[1], sc_all->nb[1] * compress_ratio, 0);
    score = ggml_add(ctx, score, ggml_repeat(ctx, ape_f, score));

    ggml_tensor * kv_comp = nullptr;
    if (coff == 1) {
        ggml_tensor * kv_p = ggml_cont(ctx, ggml_permute(ctx, kv,    1, 0, 2, 3)); // [ratio, head_dim, n_comp]
        ggml_tensor * sc_p = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));
        kv_comp = dsv4_softmax_pool_ratio(ctx, kv_p, sc_p);                        // [head_dim, n_comp]
    } else {
        ggml_tensor * kv_prev = ggml_view_3d(ctx, kv, head_dim, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], 0);
        ggml_tensor * kv_curr = ggml_view_3d(ctx, kv, head_dim, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], head_dim * kv->nb[0]);
        ggml_tensor * score_prev = ggml_view_3d(ctx, score, head_dim, compress_ratio, n_comp,
                score->nb[1], score->nb[2], 0);
        ggml_tensor * score_curr = ggml_view_3d(ctx, score, head_dim, compress_ratio, n_comp,
                score->nb[1], score->nb[2], head_dim * score->nb[0]);

        // Chunk-0 overlap comes from the carried state instead of a zero pad:
        // state rows [0, ratio) are the previous ubatch's last full window
        // (score rows already carry their APE), take their first-half slice
        // like the in-batch prev views do.
        ggml_tensor * seed_kv = ggml_view_3d(ctx, prev_kv_state, head_dim, compress_ratio, 1,
                prev_kv_state->nb[1], prev_kv_state->nb[1] * compress_ratio, 0);
        ggml_tensor * seed_sc = ggml_view_3d(ctx, prev_score_state, head_dim, compress_ratio, 1,
                prev_score_state->nb[1], prev_score_state->nb[1] * compress_ratio, 0);

        if (n_comp == 1) {
            kv_prev    = seed_kv;
            score_prev = seed_sc;
        } else {
            ggml_tensor * kv_prev_hi = ggml_view_3d(ctx, kv, head_dim, compress_ratio, n_comp - 1,
                    kv->nb[1], kv->nb[2], 0);
            ggml_tensor * sc_prev_hi = ggml_view_3d(ctx, score, head_dim, compress_ratio, n_comp - 1,
                    score->nb[1], score->nb[2], 0);
            kv_prev    = ggml_concat(ctx, seed_kv, kv_prev_hi, 2);
            score_prev = ggml_concat(ctx, seed_sc, sc_prev_hi, 2);
        }

        kv_prev    = ggml_cont(ctx, ggml_permute(ctx, kv_prev,    1, 0, 2, 3)); // [ratio, head_dim, n_comp]
        kv_curr    = ggml_cont(ctx, ggml_permute(ctx, kv_curr,    1, 0, 2, 3));
        score_prev = ggml_cont(ctx, ggml_permute(ctx, score_prev, 1, 0, 2, 3));
        score_curr = ggml_cont(ctx, ggml_permute(ctx, score_curr, 1, 0, 2, 3));

        ggml_tensor * kv_pool = ggml_concat(ctx, kv_prev,    kv_curr,    0);    // [2*ratio, head_dim, n_comp]
        ggml_tensor * sc_pool = ggml_concat(ctx, score_prev, score_curr, 0);
        kv_comp = dsv4_softmax_pool_ratio(ctx, kv_pool, sc_pool);               // [head_dim, n_comp]
    }

    kv_comp = ggml_rms_norm(ctx, kv_comp, norm_eps);
    kv_comp = ggml_mul(ctx, kv_comp, norm);
    kv_comp = ggml_reshape_3d(ctx, kv_comp, head_dim, 1, n_comp);

    // RoPE positions of the emitted rows = first token of each window.
    ggml_tensor * comp_pos = ggml_arange(ctx, (float) first_pos,
            (float) (first_pos + n_comp * compress_ratio), (float) compress_ratio);
    comp_pos = ggml_cast(ctx, comp_pos, GGML_TYPE_I32);
    kv_comp = dsv4_apply_rope_tail(ctx, kv_comp, comp_pos,
            head_dim, 1, n_comp, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

    // ---- end state (mirrors dsv4_build_compressor_prefill_state, aligned) ----
    ggml_tensor * last_kv = ggml_view_2d(ctx, kv_all, layout.width, compress_ratio,
            kv_all->nb[1], (n_tokens - compress_ratio) * kv_all->nb[1]);
    ggml_tensor * last_sc = ggml_view_2d(ctx, sc_all, layout.width, compress_ratio,
            sc_all->nb[1], (n_tokens - compress_ratio) * sc_all->nb[1]);
    last_sc = ggml_add(ctx, last_sc, ape_f);

    ggml_tensor * kv_state    = nullptr;
    ggml_tensor * score_state = nullptr;
    if (compress_ratio == 4) {
        // [prev full window | empty current partial window]
        kv_state    = ggml_concat(ctx, last_kv, dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f), 1);
        score_state = ggml_concat(ctx, last_sc, dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY), 1);
    } else {
        // ratio-128 state holds only the current partial window; aligned end -> empty.
        kv_state    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
        score_state = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);
    }

    return { kv_state, score_state, kv_comp };
}

static ggml_tensor * dsv4_build_indexer_scores_prefill(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * qr,
        ggml_tensor        * index_kv,
        ggml_tensor        * wq_b,
        ggml_tensor        * wproj,
        ggml_tensor        * pos,
        ggml_tensor        * causal_mask,
        int64_t              n_index_head,
        int64_t              n_index_head_size,
        int64_t              n_tokens,
        int64_t              n_rot,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        int32_t              skip_pos0     = -1,
        int32_t              skip_ratio    = 0) {
    ggml_tensor * q = ggml_mul_mat(ctx, wq_b, qr);
    q = ggml_reshape_3d(ctx, q, n_index_head_size, n_index_head, n_tokens);
    q = dsv4_apply_rope_tail(ctx, q, pos,
            n_index_head_size, n_index_head, n_tokens, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

    if (getenv("DSV4_NO_LIGHTNING_IDX") == nullptr) {
        // Fused NSA indexer scoring (the upstream lightning-indexer kernel,
        // also used by DeepSeek V3.2). The decomposed path below materializes
        // a [n_comp, n_tokens, n_heads] F32 intermediate - at 125K ctx that
        // blob is ~4 GB per layer and OOMed the compute buffer (the 131K
        // prefill died at depth ~73K). Math is identical: relu is positively
        // homogeneous, so the decomposed path's combined
        // 1/sqrt(head_size*n_heads) weight scale equals scale_embd*scale_heads.
        ggml_tensor * w = ggml_mul_mat(ctx, wproj, x); // [n_heads, n_tokens]
        ggml_tensor * score = ggml_lightning_indexer(ctx, q, index_kv, w,
                1.0f / std::sqrt((float) n_index_head_size),
                1.0f / std::sqrt((float) n_index_head));
        // DSV4_IDX_SKIP=1: let the CUDA kernel skip causally-invisible KV
        // blocks (n_visible = (pos + 1) / ratio, same formula as
        // fill_compress_causal - the skipped scores are written as 0.0f and
        // the causal_mask add below turns them into -INF just like before)
        static const bool idx_skip = getenv("DSV4_IDX_SKIP") != nullptr;
        if (idx_skip && skip_pos0 >= 0 && skip_ratio > 0) {
            score->op_params[2] = skip_pos0;
            score->op_params[3] = skip_ratio;
            score->op_params[4] = 1;
        }
        score = ggml_reshape_2d(ctx, score, index_kv->ne[2], n_tokens);
        return ggml_add(ctx, score, causal_mask);
    }

    ggml_tensor * k = ggml_permute(ctx, index_kv, 0, 2, 1, 3); // [head_dim, n_comp, 1]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);                     // [head_dim, n_tokens, n_heads]

    ggml_tensor * score = ggml_mul_mat(ctx, k, q);            // [n_comp, n_tokens, n_heads]
    score = ggml_relu(ctx, score);

    ggml_tensor * weights = ggml_mul_mat(ctx, wproj, x);      // [n_heads, n_tokens]
    const float scale = 1.0f / std::sqrt(float(n_index_head_size) * float(n_index_head));
    weights = dsv4_mul_scalar(ctx, weights, scale);
    weights = ggml_reshape_3d(ctx, weights, 1, n_index_head, n_tokens);
    weights = ggml_permute(ctx, weights, 0, 2, 1, 3);         // [1, n_tokens, n_heads]

    score = ggml_mul(ctx, score, weights);
    score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 2, 0, 3)); // [n_heads, n_comp, n_tokens]
    score = ggml_sum_rows(ctx, score);                            // [1, n_comp, n_tokens]
    score = ggml_reshape_2d(ctx, score, index_kv->ne[2], n_tokens);

    return ggml_add(ctx, score, causal_mask);
}

static ggml_tensor * dsv4_build_indexer_scores_decode(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * qr,
        ggml_tensor        * index_kv,
        ggml_tensor        * wq_b,
        ggml_tensor        * wproj,
        ggml_tensor        * pos,
        int64_t              n_index_head,
        int64_t              n_index_head_size,
        int64_t              n_comp,
        int64_t              n_rot,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg) {
    ggml_tensor * q = ggml_mul_mat(ctx, wq_b, qr);
    q = ggml_reshape_3d(ctx, q, n_index_head_size, n_index_head, 1);
    q = dsv4_apply_rope_tail(ctx, q, pos,
            n_index_head_size, n_index_head, 1, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

    ggml_tensor * k = ggml_reshape_3d(ctx, index_kv, n_index_head_size, 1, n_comp);
    k = ggml_permute(ctx, k, 0, 2, 1, 3); // [head_dim, n_comp, 1]
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // [head_dim, 1, n_heads]

    ggml_tensor * score = ggml_mul_mat(ctx, k, q); // [n_comp, 1, n_heads]
    score = ggml_relu(ctx, score);

    ggml_tensor * weights = ggml_mul_mat(ctx, wproj, x); // [n_heads, 1]
    const float scale = 1.0f / std::sqrt(float(n_index_head_size) * float(n_index_head));
    weights = dsv4_mul_scalar(ctx, weights, scale);
    weights = ggml_reshape_3d(ctx, weights, 1, n_index_head, 1);
    weights = ggml_permute(ctx, weights, 0, 2, 1, 3); // [1, 1, n_heads]

    score = ggml_mul(ctx, score, weights);
    score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 2, 0, 3)); // [n_heads, n_comp, 1]
    score = ggml_sum_rows(ctx, score);
    return ggml_reshape_2d(ctx, score, n_comp, 1);
}

static ggml_tensor * dsv4_build_compressed_mask_from_topk(
        ggml_context * ctx,
        ggml_tensor  * scores,
        ggml_tensor  * topk) {
    const int64_t n_comp   = scores->ne[0];
    const int64_t n_tokens = scores->ne[1];

    ggml_tensor * scores_rows = ggml_reshape_3d(ctx, scores, 1, scores->ne[0], scores->ne[1]);
    ggml_tensor * selected_scores = ggml_get_rows(ctx, scores_rows, topk); // [1, top_k, n_tokens]
    ggml_tensor * valid = ggml_step(ctx, dsv4_add_scalar(ctx, selected_scores, 1.0e30f));
    ggml_tensor * values = dsv4_mul_scalar(ctx, dsv4_add_scalar(ctx, valid, -1.0f), 1.0e9f);

    ggml_tensor * mask = dsv4_new_filled_3d(ctx, 1, n_comp, n_tokens, -INFINITY);
    mask = ggml_set_rows(ctx, mask, values, topk);
    return ggml_reshape_2d(ctx, mask, n_comp, n_tokens);
}

static ggml_tensor * dsv4_cache_view_3d(ggml_context * ctx, ggml_tensor * cache, int64_t n_rows) {
    ggml_tensor * view = ggml_view_2d(ctx, cache, cache->ne[0], n_rows, cache->nb[1], 0);
    return ggml_reshape_3d(ctx, view, cache->ne[0], 1, n_rows);
}

// ---------------------------------------------------------------------------
// MTP draft module (phase A: offline acceptance testing).
//
// The 32 mtp.0.* tensors ship in a SEPARATE gguf (antirez's split); they form
// one full DS4 layer (window MLA, no NSA) plus e_proj/h_proj input fusion and
// an hc_head collapse sharing the base model's output.weight. We side-load
// them outside llama_model_loader into a standalone backend buffer
// (DSV4_MTP_DEV, default CUDA3 — same device as model.output under layer
// split, so the whole draft branch stays on one GPU).
// ---------------------------------------------------------------------------

// Speculative-decode ring size: matches the MTP layer's training-time SWA
// window. Ring rows hold the fp8-quantized MTP-layer KV of the last 128
// accepted (or teacher-forced) positions; ring_pos_cpu[r] is the absolute
// position stored in row r (-1 = empty). The CLIENT owns ring_pos_cpu updates
// (dsv4_mtp_spec_commit) so graph warmup/reserve runs never corrupt it.
static constexpr int64_t DSV4_MTP_RING = 128;

struct dsv4_mtp_module {
    bool attempted = false;
    bool ok        = false;
    ggml_context         * wctx = nullptr;
    ggml_backend_buffer_t  wbuf = nullptr;
    ggml_backend_buffer_type_t wbuft = nullptr;
    std::map<std::string, ggml_tensor *> tensors;

    // phase-B speculative state
    ggml_context         * rctx = nullptr;   // ring tensor ctx
    ggml_backend_buffer_t  rbuf = nullptr;
    ggml_tensor          * ring_kv = nullptr; // [kv_w, DSV4_MTP_RING], fp8-kv type
    ggml_tensor          * dout = nullptr;    // i32 [2]: draft tokens (static output)
    ggml_tensor          * hout = nullptr;    // f32 [n_embd*n_hc*2]: hc_hist (static output)
    int64_t   hc_elems = 0;                   // n_embd*n_hc
    int64_t   ring_pos_cpu[DSV4_MTP_RING];    // abs position per row, -1 empty
    std::vector<float> hc_seed_cpu;           // [n_embd*n_hc] HC of the token before the batch

    // Device-resident mirror of the base model's token embedding (lazy).
    // get_rows(tok_embd, argmax-i32) with the host-buffer embedding lands on
    // CPU and drags the i32 draft id through a cross-backend copy that is
    // unreliable in this build (stale/garbage id -> OOB row assert; hit on
    // genuine decode after a checkpoint restore, not just the prefill
    // boundary that 859e602 gated off). Mirroring the embedding onto the MTP
    // device keeps the whole candidate lookup on-GPU.
    ggml_context         * ectx = nullptr;
    ggml_backend_buffer_t  ebuf = nullptr;
    ggml_tensor          * tok_embd_dev = nullptr;
    bool embd_attempted = false;

    ggml_tensor * t(const char * name) const {
        auto it = tensors.find(name);
        GGML_ASSERT(it != tensors.end() && "dsv4-mtp: missing tensor");
        return it->second;
    }

    // Lazily create the ring buffer once the kv row layout is known (first
    // spec-branch build). Zero-filled: unwritten rows are masked out by
    // ring_pos_cpu < 0, but FA still multiplies masked K rows by Q before the
    // -inf mask lands, and uninitialized memory could be NaN (0*NaN = NaN).
    bool ensure_ring(ggml_type kv_type, int64_t kv_w, int64_t n_embd, int64_t n_hc) {
        if (ring_kv != nullptr) {
            return true;
        }
        ggml_init_params rp = { 4 * ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
        rctx = ggml_init(rp);
        ring_kv = ggml_new_tensor_2d(rctx, kv_type, kv_w, DSV4_MTP_RING);
        ggml_set_name(ring_kv, "dsv4_mtp_ring_kv");
        hc_elems = n_embd * n_hc;
        // F32, not I32: CUDA has no I32->I32 CPY kernel, so an I32 static copy
        // silently falls back to CPU and the device buffer never gets written
        // (the draft reads were garbage). Cast to F32 in-graph instead; token
        // ids up to ~2^24 are exact in F32.
        dout = ggml_new_tensor_1d(rctx, GGML_TYPE_F32, 4);
        ggml_set_name(dout, "dsv4_mtp_dout");
        hout = ggml_new_tensor_1d(rctx, GGML_TYPE_F32, hc_elems * 3);
        ggml_set_name(hout, "dsv4_mtp_hout");
        rbuf = ggml_backend_alloc_ctx_tensors_from_buft(rctx, wbuft);
        if (rbuf == nullptr) {
            fprintf(stderr, "dsv4-mtp: ring alloc failed\n");
            ring_kv = nullptr;
            return false;
        }
        ggml_backend_buffer_clear(rbuf, 0);
        for (int64_t i = 0; i < DSV4_MTP_RING; ++i) {
            ring_pos_cpu[i] = -1;
        }
        return true;
    }

    // Device mirror of the token embedding, or nullptr (caller falls back to
    // the host tensor). One-shot: an alloc failure is not retried.
    ggml_tensor * ensure_tok_embd(ggml_tensor * src) {
        if (embd_attempted) {
            return tok_embd_dev;
        }
        embd_attempted = true;
        if (src == nullptr || wbuft == nullptr) {
            return nullptr;
        }
        if (src->buffer == nullptr || src->data == nullptr) {
            // memory-estimation pass builds graphs before weights are loaded;
            // retry once the real tensor data exists
            embd_attempted = false;
            return nullptr;
        }
        // DSV4_MTP_EMBD_DEV moves the ~1 GiB mirror off the MTP device (VRAM
        // valve): the get_rows node follows the weight, its i32 index and the
        // resulting embedding travel device-to-device, which is the reliable
        // path (only the CPU-backend hop was flaky).
        ggml_backend_buffer_type_t ebuft = wbuft;
        if (const char * edev = getenv("DSV4_MTP_EMBD_DEV")) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(edev);
            if (dev != nullptr) {
                ebuft = ggml_backend_dev_buffer_type(dev);
            } else {
                fprintf(stderr, "dsv4-mtp: DSV4_MTP_EMBD_DEV=%s not found, using MTP device\n", edev);
            }
        }
        ggml_init_params ep = { 2 * ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
        ectx = ggml_init(ep);
        ggml_tensor * dst = ggml_dup_tensor(ectx, src);
        ggml_set_name(dst, "dsv4_mtp_tok_embd_dev");
        ebuf = ggml_backend_alloc_ctx_tensors_from_buft(ectx, ebuft);
        if (ebuf == nullptr) {
            fprintf(stderr, "dsv4-mtp: tok_embd device mirror alloc failed, keeping host path\n");
            return nullptr;
        }
        ggml_backend_buffer_set_usage(ebuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const size_t nb   = ggml_nbytes(src);
        const size_t step = 64ull * 1024 * 1024;
        std::vector<uint8_t> tmp(nb < step ? nb : step);
        for (size_t off = 0; off < nb; off += step) {
            const size_t n = nb - off < step ? nb - off : step;
            ggml_backend_tensor_get(src, tmp.data(), off, n);
            ggml_backend_tensor_set(dst, tmp.data(), off, n);
        }
        tok_embd_dev = dst;
        fprintf(stderr, "dsv4-mtp: token embedding mirrored onto MTP device (%zu MiB)\n", nb >> 20);
        return dst;
    }
};

static dsv4_mtp_module & dsv4_mtp_get() {
    static dsv4_mtp_module m;
    if (m.attempted) {
        return m;
    }
    m.attempted = true;
    const char * path = getenv("DSV4_MTP_GGUF");
    if (path == nullptr) {
        return m;
    }

    ggml_context * meta = nullptr;
    gguf_init_params ip = { /*no_alloc*/ true, /*ctx*/ &meta };
    gguf_context * g = gguf_init_from_file(path, ip);
    if (g == nullptr) {
        fprintf(stderr, "dsv4-mtp: failed to open %s\n", path);
        return m;
    }

    const int64_t n_t = gguf_get_n_tensors(g);
    ggml_init_params wp = { (size_t)(n_t + 2) * ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
    m.wctx = ggml_init(wp);

    std::vector<std::pair<ggml_tensor *, size_t>> pending;
    for (int64_t i = 0; i < n_t; ++i) {
        const char  * nm  = gguf_get_tensor_name(g, i);
        ggml_tensor * src = ggml_get_tensor(meta, nm);
        ggml_tensor * dst = ggml_dup_tensor(m.wctx, src);
        ggml_set_name(dst, nm);
        pending.emplace_back(dst, (size_t)gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i));
        m.tensors[nm] = dst;
    }

    const char * devname = getenv("DSV4_MTP_DEV") ? getenv("DSV4_MTP_DEV") : "CUDA3";
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(devname);
    ggml_backend_buffer_type_t buft = dev != nullptr
        ? ggml_backend_dev_buffer_type(dev)
        : ggml_backend_cpu_buffer_type();
    m.wbuft = buft;
    m.wbuf = ggml_backend_alloc_ctx_tensors_from_buft(m.wctx, buft);
    if (m.wbuf == nullptr && buft != ggml_backend_cpu_buffer_type()) {
        fprintf(stderr, "dsv4-mtp: alloc on %s failed, falling back to CPU\n", devname);
        m.wbuf = ggml_backend_alloc_ctx_tensors_from_buft(m.wctx, ggml_backend_cpu_buffer_type());
    }
    if (m.wbuf == nullptr) {
        fprintf(stderr, "dsv4-mtp: buffer alloc failed\n");
        gguf_free(g);
        ggml_free(meta);
        return m;
    }
    ggml_backend_buffer_set_usage(m.wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * f = fopen(path, "rb");
    if (f == nullptr) {
        fprintf(stderr, "dsv4-mtp: reopen failed for %s\n", path);
        gguf_free(g);
        ggml_free(meta);
        return m;
    }
    std::vector<uint8_t> tmp;
    for (auto & pr : pending) {
        const size_t nb = ggml_nbytes(pr.first);
        tmp.resize(nb);
        if (fseek(f, (long)pr.second, SEEK_SET) != 0 || fread(tmp.data(), 1, nb, f) != nb) {
            fprintf(stderr, "dsv4-mtp: short read on %s\n", ggml_get_name(pr.first));
            fclose(f);
            gguf_free(g);
            ggml_free(meta);
            return m;
        }
        ggml_backend_tensor_set(pr.first, tmp.data(), 0, nb);
    }
    fclose(f);
    gguf_free(g);
    ggml_free(meta);

    m.ok = true;
    fprintf(stderr, "dsv4-mtp: loaded %lld tensors from %s onto %s\n",
            (long long)n_t, path, dev != nullptr ? devname : "CPU");
    return m;
}

static int64_t dsv4_mtp_ring_pos_at(int64_t r) {
    dsv4_mtp_module & m = dsv4_mtp_get();
    if (m.ring_kv == nullptr || r < 0 || r >= DSV4_MTP_RING) {
        return -1;
    }
    return m.ring_pos_cpu[r];
}

static const float * dsv4_mtp_seed_data(size_t need) {
    dsv4_mtp_module & m = dsv4_mtp_get();
    if (m.hc_seed_cpu.size() < need) {
        m.hc_seed_cpu.resize(need, 0.0f);
    }
    return m.hc_seed_cpu.data();
}

// Causal sliding-window mask [n_kv=n_tokens, n_q=n_tokens] over the current
// ubatch: visible iff 0 <= q_idx - k_idx < window. Built on-device from
// arange so no CPU fill / no new graph input is needed.
static ggml_tensor * dsv4_gpu_mask_raw_swa(
        ggml_context * ctx,
        int64_t        n_tokens,
        int64_t        window) {
    ggml_tensor * ar_k = ggml_arange(ctx, 0.0f, (float) n_tokens - 0.5f, 1.0f);   // [nt]
    ar_k = dsv4_mul_scalar(ctx, ar_k, -1.0f);
    ar_k = ggml_repeat_4d(ctx, ggml_reshape_2d(ctx, ar_k, n_tokens, 1), n_tokens, n_tokens, 1, 1);
    ggml_tensor * ar_q = ggml_arange(ctx, 0.0f, (float) n_tokens - 0.5f, 1.0f);
    ar_q = ggml_reshape_2d(ctx, ar_q, 1, n_tokens);
    ggml_tensor * d = ggml_add(ctx, ar_k, ar_q);                                  // d[k,q] = q - k
    ggml_tensor * vis  = ggml_step(ctx, dsv4_add_scalar(ctx, d, 0.5f));            // q >= k
    ggml_tensor * vis2 = ggml_step(ctx, dsv4_add_scalar(ctx,
            dsv4_mul_scalar(ctx, d, -1.0f), (float) window - 0.5f));               // q - k < window
    vis = ggml_mul(ctx, vis, vis2);
    return dsv4_mul_scalar(ctx, dsv4_add_scalar(ctx, vis, -1.0f), 1.0e9f);         // 0 / -1e9
}

} // namespace

llama_model_deepseek4::graph::graph(const llama_model & model, const llm_graph_params & params) :
	llm_graph_context(params) {

    const int64_t n_hc        = hparams.n_hc;
    const int64_t n_lora_q    = hparams.n_lora_q;
    const int64_t n_lora_o    = hparams.n_lora_o;
    const int64_t n_out_group = hparams.n_attn_out_groups;

    GGML_ASSERT(n_hc > 0);
    GGML_ASSERT(n_lora_q > 0);
    GGML_ASSERT(n_lora_o > 0);
    GGML_ASSERT(n_out_group > 0);
    GGML_ASSERT(n_embd_head_k == n_embd_head_v);
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_tokens = res->t_inp_tokens;
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Raw token embeddings [n_embd, n_tokens] for the MTP draft branch (e-path).
    ggml_tensor * mtp_emb0 = inpL;

    auto * inp_mem  = build_inp_mem_hybrid_iswa();
    auto * inp_attn = inp_mem->get_attn();
    auto * inp_rs   = inp_mem->get_recr();
    const auto * mctx_dsv4 = inp_mem->mctx;
    dsv4_graph_inputs * inp_dsv4 = nullptr;
    auto get_dsv4_inputs = [&]() {
        if (inp_dsv4 == nullptr) {
            auto inputs = std::make_unique<dsv4_graph_inputs>();
            inp_dsv4 = inputs.get();
            res->add_input(std::move(inputs));
        }
        return inp_dsv4;
    };

    // Cache of GPU-built causal comp masks, shared across layers of the same
    // ratio/width (all ratio-4 layers see one mask, all ratio-128 another).
    std::vector<std::tuple<int64_t, int64_t, ggml_tensor *>> gpu_comp_masks;
    auto get_gpu_comp_mask = [&](int64_t width, int64_t ratio) -> ggml_tensor * {
        for (const auto & [r, w, t] : gpu_comp_masks) {
            if (r == ratio && w == width) {
                return t;
            }
        }
        ggml_tensor * m = dsv4_gpu_mask_comp_causal(ctx0, inp_pos, width, ratio, n_tokens);
        gpu_comp_masks.emplace_back(ratio, width, m);
        return m;
    };

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);

    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head_k));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        const uint32_t compress_ratio = hparams.attn_compress_ratio[il];
        const dsv4_rope_cfg rope_cfg = dsv4_make_rope_cfg(hparams, cparams, compress_ratio);
        const bool is_prefill = ubatch.pos == nullptr || ubatch.pos[0] == 0;

        if (compress_ratio != 0) {
            if (compress_ratio != 4 && compress_ratio != 128) {
                throw std::runtime_error("DeepSeek V4 unsupported attention compression ratio " + std::to_string(compress_ratio));
            }
            // The hybrid memory splitter emits one sequence set per ubatch
            // for compressed DeepSeek V4 attention.
            GGML_ASSERT(ubatch.n_seqs == 1);
        }

        ggml_tensor * residual = inpL;
        dsv4_hc_mix mix = dsv4_hc_pre(ctx0, inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        ggml_tensor * cur = mix.x;
        cb(cur, "hc_attn_pre", il);
        cb(mix.mixes, "hc_attn_pre_mixes", il);
        cb(mix.pre, "hc_attn_pre_weights", il);
        cb(mix.post, "hc_attn_pre_post_weights", il);
        cb(mix.comb, "hc_attn_pre_comb", il);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(qr, "q_lora", il);
        qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(qr, "q_lora_norm", il);

        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
        q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
        q = ggml_rms_norm(ctx0, q, norm_rms_eps);
        cb(q, "Qnorm", il);
        q = dsv4_apply_rope_tail(ctx0, q, inp_pos,
                n_embd_head_k, n_head, n_tokens, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
        cb(q, "Qcur", il);
        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.attn_kv, cur);
        kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head_k, 1, n_tokens);
        cb(kv, "KVnorm", il);
        kv = dsv4_apply_rope_tail(ctx0, kv, inp_pos,
                n_embd_head_k, 1, n_tokens, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
        cb(kv, "KVrope", il);
        kv = ggml_dsv4_fp8_kv_quantize(ctx0, kv, n_rot);
        cb(kv, "KVcur", il);

        const auto * mctx_swa = inp_attn->mctx->get_swa();
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, kv);
        ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));

        if (compress_ratio == 0) {
            ggml_tensor * k_cache = mctx_swa->get_k(ctx0, il);
            k_cache = ggml_reshape_3d(ctx0, k_cache, n_embd_head_k, 1, k_cache->ne[2]);
            cur = build_attn_mha(q, k_cache, k_cache, nullptr, inp_attn->get_kq_mask_swa(),
                    layer.attn_sinks, nullptr, nullptr, kq_scale, il);
            cb(cur, "kqv_out", il);
        } else {
            ggml_tensor * k_all = kv;
            ggml_tensor * v_all = kv;
            ggml_tensor * attn_mask = nullptr;
            const llama_seq_id seq_id = ubatch.seq_id[0][0];
            // Return the post-store cache for the READ sequence (seq_id) so the
            // caller can thread it into the cache read -> explicit store->read edge.
            auto store_attn_cache_rows = [&](ggml_tensor * src, int64_t row_start, int64_t n_rows, ggml_tensor * row_idx) -> ggml_tensor * {
                ggml_tensor * updated_for_read = nullptr;
                for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                    const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                    ggml_tensor * u = dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_attn_k(ctx0, il, dst_seq_id), src, row_start, n_rows, row_idx);
                    if (dst_seq_id == seq_id) updated_for_read = u;
                }
                return updated_for_read;
            };
            auto store_index_cache_rows = [&](ggml_tensor * src, int64_t row_start, int64_t n_rows, ggml_tensor * row_idx) -> ggml_tensor * {
                ggml_tensor * updated_for_read = nullptr;
                for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                    const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                    ggml_tensor * u = dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_index_k(ctx0, il, dst_seq_id), src, row_start, n_rows, row_idx);
                    if (dst_seq_id == seq_id) updated_for_read = u;
                }
                return updated_for_read;
            };
            const int64_t state_size = hparams.n_embd_r();
            const dsv4_state_layout attn_state_layout = dsv4_make_state_layout(compress_ratio, n_embd_head_k);

            ggml_tensor * prev_kv_state_all = build_rs(inp_rs, inp_rs->mctx->get_r_l(il), state_size, ubatch.n_seqs);
            ggml_tensor * prev_sc_state_all = build_rs(inp_rs, inp_rs->mctx->get_s_l(il), state_size, ubatch.n_seqs);
            ggml_tensor * prev_attn_kv_state = dsv4_view_state_segment(ctx0, prev_kv_state_all, 0, attn_state_layout.width, attn_state_layout.rows);
            ggml_tensor * prev_attn_sc_state = dsv4_view_state_segment(ctx0, prev_sc_state_all, 0, attn_state_layout.width, attn_state_layout.rows);

            const int64_t n_comp = n_tokens / compress_ratio;
            if (is_prefill) {
                dsv4_state_pair state = dsv4_build_compressor_prefill_state(ctx0, cur,
                        layer.attn_compressor_kv,
                        layer.attn_compressor_gate,
                        layer.attn_compressor_ape,
                        n_embd_head_k,
                        n_tokens,
                        compress_ratio);
                dsv4_store_state_segment(ctx0, gf, state.kv,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0);
                dsv4_store_state_segment(ctx0, gf, state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0);

                if (compress_ratio == 4) {
                    const dsv4_state_layout index_state_layout = dsv4_make_state_layout(compress_ratio, hparams.indexer_head_size);
                    dsv4_state_pair index_state = dsv4_build_compressor_prefill_state(ctx0, cur,
                            layer.indexer_compressor_kv,
                            layer.indexer_compressor_gate,
                            layer.indexer_compressor_ape,
                            hparams.indexer_head_size,
                            n_tokens,
                            compress_ratio);
                    dsv4_store_state_segment(ctx0, gf, index_state.kv,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                    dsv4_store_state_segment(ctx0, gf, index_state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                    GGML_ASSERT(attn_state_layout.elems + index_state_layout.elems <= state_size);
                }
            }

            if (is_prefill && n_comp > 0) {
                ggml_tensor * comp_pos = ggml_arange(ctx0, 0.0f, float(n_comp * compress_ratio), float(compress_ratio));
                comp_pos = ggml_cast(ctx0, comp_pos, GGML_TYPE_I32);

                ggml_tensor * kv_comp = dsv4_build_compressor_prefill(ctx0, cur,
                        layer.attn_compressor_kv,
                        layer.attn_compressor_gate,
                        layer.attn_compressor_ape,
                        layer.attn_compressor_norm,
                        comp_pos,
                        n_embd_head_k, n_rot, n_tokens, compress_ratio, rope_type, rope_cfg, norm_rms_eps);
                kv_comp = ggml_dsv4_fp8_kv_quantize(ctx0, kv_comp, n_rot);
                cb(kv_comp, "KVcompress", il);

                store_attn_cache_rows(kv_comp, 0, n_comp, nullptr);

                k_all = ggml_concat(ctx0, kv, kv_comp, 2);
                v_all = k_all;

                if (compress_ratio == 4) {
                    ggml_tensor * raw_mask = get_dsv4_inputs()->add_mask(ctx0,
                            dsv4_mask_kind::RAW_WINDOW,
                            n_tokens, n_tokens,
                            n_tokens, n_comp, hparams.n_swa, compress_ratio,
                            "dsv4_attn_raw_window_mask");
                    ggml_tensor * index_mask = get_dsv4_inputs()->add_mask(ctx0,
                            dsv4_mask_kind::COMPRESS_CAUSAL,
                            n_comp, n_tokens,
                            0, n_comp, 0, compress_ratio,
                            "dsv4_indexer_causal_mask");

                    ggml_tensor * index_kv = dsv4_build_compressor_prefill(ctx0, cur,
                            layer.indexer_compressor_kv,
                            layer.indexer_compressor_gate,
                            layer.indexer_compressor_ape,
                            layer.indexer_compressor_norm,
                            comp_pos,
                            hparams.indexer_head_size, n_rot, n_tokens, compress_ratio, rope_type, rope_cfg, norm_rms_eps);
                    cb(index_kv, "indexer_KVcompress", il);

                    store_index_cache_rows(index_kv, 0, n_comp, nullptr);

                    ggml_tensor * index_scores = dsv4_build_indexer_scores_prefill(ctx0,
                            cur, qr, index_kv,
                            layer.indexer_attn_q_b,
                            layer.indexer_proj,
                            inp_pos,
                            index_mask,
                            hparams.indexer_n_head,
                            hparams.indexer_head_size,
                            n_tokens,
                            n_rot,
                            rope_type,
                            rope_cfg,
                            ubatch.pos ? (int32_t) ubatch.pos[0] : (int32_t) -1,
                            (int32_t) compress_ratio);
                    cb(index_scores, "indexer_scores", il);

                    const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp);
                    ggml_tensor * topk = ggml_argsort_top_k(ctx0, index_scores, top_k);
                    cb(topk, "indexer_topk", il);

                    ggml_tensor * comp_mask = dsv4_build_compressed_mask_from_topk(ctx0, index_scores, topk);
                    cb(comp_mask, "dsv4_attn_compress_mask", il);

                    attn_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
                } else {
                    attn_mask = get_dsv4_inputs()->add_mask(ctx0,
                            dsv4_mask_kind::ATTN_STATIC,
                            n_tokens + n_comp, n_tokens,
                            n_tokens, n_comp, hparams.n_swa, compress_ratio,
                            "dsv4_attn_static_mask");
                }
            } else {
                attn_mask = get_dsv4_inputs()->add_mask(ctx0,
                        dsv4_mask_kind::RAW_WINDOW,
                        n_tokens, n_tokens,
                        n_tokens, 0, hparams.n_swa, compress_ratio,
                        "dsv4_attn_raw_window_mask");
            }

            // set by the unrolled const-shape verify-pair path (nt==2) below;
            // when non-null it IS the attention output and the shared pad+mha
            // tail is skipped.
            ggml_tensor * cs_pair_out = nullptr;

            // set by the sparse top-k FA prompt-chunk path (DSV4_SPARSE_FA):
            // per-token int32 index list handed to flash-attn as src[5]
            ggml_tensor * sparse_topk = nullptr;

            if (!is_prefill) {
                const llama_pos first_pos = ubatch.pos ? ubatch.pos[0] : 0;
                const llama_pos last_pos  = ubatch.pos ? ubatch.pos[n_tokens - 1] : n_tokens - 1;
                const int64_t n_comp_before  = first_pos / compress_ratio;
                const int64_t n_comp_visible = (last_pos + 1) / compress_ratio;
                const int64_t n_comp_cache = mctx_dsv4->get_dsv4_n_comp(il);
                GGML_ASSERT(n_comp_visible <= n_comp_cache);

                // Force a rebuild when n_comp_visible crosses 0 for this ratio:
                // a graph built without the comp branch must not be reused once
                // compressed rows become visible (else these layers silently
                // never attend their compressed cache).
                get_dsv4_inputs()->note_comp_presence(compress_ratio, n_comp_visible > 0);

                // Constant-shape decode (env DSV4_CONSTANT_SHAPE): shared by the
                // attn+index compressors AND the attention cache view/mask below.
                // When enabled, compressors ALWAYS-build + blend the state shift on
                // real boundaries (per cs_boundary), use cs_comp_pos for the emitted
                // chunk's RoPE position, and the cache view/mask use a fixed width
                // -> the whole decode graph is depth-invariant (CUDA-graph replay
                // at all depths). Ratio-4, single-token, <=top_k regime only.
                const int64_t cs_nc_fixed = std::min<int64_t>(hparams.indexer_top_k, n_comp_cache);
                // Require cs_nc_fixed < n_comp_cache so the off-boundary scratch row
                // (n_comp_cache-1) is OUTSIDE the attended fixed view [0, cs_nc_fixed)
                // -> discarded partial writes are never read by attention.
                const bool cs_infra_ok = dsv4_constant_shape_enabled()
                    && compress_ratio == 4 && n_tokens == 1
                    && cs_nc_fixed < n_comp_cache;
                // Fixed-mask regime: everything visible fits under top_k, attend
                // the fixed [0, cs_nc_fixed) view with a COMPRESS_FIXED mask.
                const bool use_const_shape = cs_infra_ok && n_comp_visible <= cs_nc_fixed;
                // Beyond top_k the top-k branch selects 512 rows anyway: GATHER
                // them (get_rows) instead of masking the full growing width, so
                // the attention shape stays [n_raw + top_k] at ANY depth and the
                // indexer score width is bucketed (shape changes only every
                // DSV4_CS_BUCKET*ratio tokens) -> the decode graph replays at
                // long context too instead of collapsing into recapture.
                const bool use_topk_gather = cs_infra_ok && n_comp_visible > cs_nc_fixed
                    && !dsv4_topk_gather_disabled();
                // The always-build compressor machinery (blend + input-driven
                // positions/rows) is required by BOTH constant-shape regimes.
                const bool cs_compressor = use_const_shape || use_topk_gather;
                ggml_tensor * cs_comp_pos = cs_compressor
                    ? get_dsv4_inputs()->add_comp_pos_index(ctx0, compress_ratio, "dsv4_cs_comp_pos") : nullptr;
                ggml_tensor * cs_boundary = cs_compressor
                    ? get_dsv4_inputs()->add_boundary_flag(ctx0, compress_ratio, "dsv4_cs_boundary") : nullptr;
                // Cache write row for the always-built store: real completed-chunk
                // row on a boundary, else a scratch row (last cache row, never
                // attended since NC_FIXED < n_comp_cache) so off-boundary writes
                // of the partial compressed row are discarded.
                ggml_tensor * cs_comp_row = cs_compressor
                    ? get_dsv4_inputs()->add_comp_row_index(ctx0, compress_ratio, n_comp_cache - 1, "dsv4_cs_comp_row") : nullptr;

                // Ratio-aligned prompt chunks take the BATCHED compressor (one
                // pooled subgraph for the whole ubatch) instead of the per-token
                // loop; unaligned tail chunks fall back to the loop.
                const bool chunk_aligned = n_tokens > 1 && ubatch.pos != nullptr
                    && first_pos % compress_ratio == 0
                    && n_tokens % compress_ratio == 0
                    && !dsv4_batched_chunk_disabled();

                // ---- Unrolled constant-shape verify PAIR (n_tokens == 2) ----
                // Speculative verify decodes are [accepted, draft] pairs. Run
                // the proven nt=1 const-shape machinery TWICE inside one graph:
                // per-token scalar inputs (off = -1 for token 0, 0 for token 1),
                // compressor state chained token0 -> token1, one attention per
                // token over column views. All shapes are depth-invariant, so
                // the verify graph reuses + CUDA-graph replays like nt=1 decode.
                const bool cs_pair = dsv4_constant_shape_enabled()
                    && compress_ratio == 4 && (n_tokens == 2 || n_tokens == 3)
                    && cs_nc_fixed < n_comp_cache
                    && n_comp_visible > 0
                    && ubatch.pos != nullptr
                    && !dsv4_topk_gather_disabled();
                if (cs_pair) {
                    auto ccat = [&](ggml_tensor * a, ggml_tensor * b, int d, const char * tag) {
                        if (a->type != b->type) {
                            fprintf(stderr, "cs_pair concat type mismatch [%s]: %s vs %s\n",
                                    tag, ggml_type_name(a->type), ggml_type_name(b->type));
                        }
                        return ggml_concat(ctx0, a, b, d);
                    };
                    const dsv4_state_layout index_state_layout =
                        dsv4_make_state_layout(compress_ratio, hparams.indexer_head_size);
                    ggml_tensor * prev_akv = prev_attn_kv_state;
                    ggml_tensor * prev_asc = prev_attn_sc_state;
                    ggml_tensor * prev_ikv = dsv4_view_state_segment(ctx0, prev_kv_state_all,
                            attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);
                    ggml_tensor * prev_isc = dsv4_view_state_segment(ctx0, prev_sc_state_all,
                            attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);

                    // regime keyed to last_pos; per-row mask content keeps
                    // token 0 honest when the pair straddles a bucket edge
                    const bool pair_gather = n_comp_visible > cs_nc_fixed;
                    const int64_t top_k  = hparams.indexer_top_k;
                    const int64_t nc_idx = pair_gather
                        ? std::min<int64_t>(GGML_PAD(n_comp_visible, DSV4_CS_BUCKET), n_comp_cache - 1)
                        : cs_nc_fixed;
                    ggml_tensor * fixed_mask = pair_gather ? nullptr
                        : get_dsv4_inputs()->add_mask(ctx0, dsv4_mask_kind::COMPRESS_FIXED,
                                cs_nc_fixed, n_tokens, 0, n_comp_visible, 0, compress_ratio,
                                "dsv4_attn_compress_mask_pair");
                    ggml_tensor * bucket_mask = pair_gather
                        ? get_dsv4_inputs()->add_mask(ctx0, dsv4_mask_kind::COMPRESS_BUCKET,
                                nc_idx, n_tokens, 0, n_comp_cache - 1, DSV4_CS_BUCKET, compress_ratio,
                                "dsv4_idx_score_bucket_mask_pair")
                        : nullptr;
                    ggml_tensor * raw_mask_pair = inp_attn->get_kq_mask_swa();
                    ggml_tensor * k_raw_pair = mctx_swa->get_k(ctx0, il);
                    k_raw_pair = ggml_reshape_3d(ctx0, k_raw_pair, n_embd_head_k, 1, k_raw_pair->ne[2]);

                    ggml_tensor * cur_c[3], * qr_c[3], * q_c[3], * pos_c[3];
                    ggml_tensor * attn_read = nullptr, * index_read = nullptr;
                    for (int t = 0; t < (int) n_tokens; ++t) {
                        const int64_t off = (int64_t) t - (n_tokens - 1);
                        cur_c[t] = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd, 1,
                                cur->nb[1], (size_t) t * cur->nb[1]));
                        qr_c[t]  = ggml_cont(ctx0, ggml_view_2d(ctx0, qr, qr->ne[0], 1,
                                qr->nb[1], (size_t) t * qr->nb[1]));
                        q_c[t]   = ggml_cont(ctx0, ggml_view_3d(ctx0, q, q->ne[0], q->ne[1], 1,
                                q->nb[1], q->nb[2], (size_t) t * q->nb[2]));
                        pos_c[t] = ggml_view_1d(ctx0, inp_pos, 1, (size_t) t * sizeof(int32_t));

                        char nb0[96];
                        auto nm = [&](const char * s) -> const char * {
                            snprintf(nb0, sizeof(nb0), "%s_p%d", s, t);
                            return nb0;
                        };
                        ggml_tensor * ape_i  = get_dsv4_inputs()->add_index(ctx0, compress_ratio, 0, nm("dsv4_attn_ape_idx"), off);
                        ggml_tensor * row_i  = get_dsv4_inputs()->add_index(ctx0, compress_ratio, compress_ratio, nm("dsv4_attn_row_idx"), off);
                        ggml_tensor * cpos_i = get_dsv4_inputs()->add_comp_pos_index(ctx0, compress_ratio, nm("dsv4_cs_comp_pos"), off);
                        ggml_tensor * bnd_i  = get_dsv4_inputs()->add_boundary_flag(ctx0, compress_ratio, nm("dsv4_cs_boundary"), off);
                        ggml_tensor * crow_i = get_dsv4_inputs()->add_comp_row_index(ctx0, compress_ratio, n_comp_cache - 1, nm("dsv4_cs_comp_row"), off);

                        dsv4_decode_compressor dec_t = dsv4_build_compressor_decode(ctx0, cur_c[t],
                                prev_akv, prev_asc,
                                layer.attn_compressor_kv,
                                layer.attn_compressor_gate,
                                layer.attn_compressor_ape,
                                layer.attn_compressor_norm,
                                n_embd_head_k, n_rot,
                                first_pos + t, compress_ratio, rope_type, rope_cfg, norm_rms_eps,
                                ape_i, row_i, cpos_i, bnd_i);
                        prev_akv = dec_t.kv_state;
                        prev_asc = dec_t.score_state;
                        if (dec_t.kv_comp != nullptr) {
                            ggml_tensor * kvq = ggml_dsv4_fp8_kv_quantize(ctx0, dec_t.kv_comp, n_rot);
                            attn_read = store_attn_cache_rows(kvq, n_comp_before,
                                    n_comp_visible - n_comp_before, crow_i);
                        }

                        ggml_tensor * iape_i = get_dsv4_inputs()->add_index(ctx0, compress_ratio, 0, nm("dsv4_idx_ape_idx"), off);
                        ggml_tensor * irow_i = get_dsv4_inputs()->add_index(ctx0, compress_ratio, compress_ratio, nm("dsv4_idx_row_idx"), off);
                        dsv4_decode_compressor idec_t = dsv4_build_compressor_decode(ctx0, cur_c[t],
                                prev_ikv, prev_isc,
                                layer.indexer_compressor_kv,
                                layer.indexer_compressor_gate,
                                layer.indexer_compressor_ape,
                                layer.indexer_compressor_norm,
                                hparams.indexer_head_size, n_rot,
                                first_pos + t, compress_ratio, rope_type, rope_cfg, norm_rms_eps,
                                iape_i, irow_i, cpos_i, bnd_i);
                        prev_ikv = idec_t.kv_state;
                        prev_isc = idec_t.score_state;
                        if (idec_t.kv_comp != nullptr) {
                            index_read = store_index_cache_rows(idec_t.kv_comp, n_comp_before,
                                    n_comp_visible - n_comp_before, crow_i);
                        }
                    }

                    // chained states after token 1 are the layer's new state
                    dsv4_store_state_segment(ctx0, gf, prev_akv, inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0);
                    dsv4_store_state_segment(ctx0, gf, prev_asc, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0);
                    dsv4_store_state_segment(ctx0, gf, prev_ikv, inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                    dsv4_store_state_segment(ctx0, gf, prev_isc, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems);

                    if (attn_read == nullptr) {
                        attn_read = mctx_dsv4->get_dsv4_attn_k(ctx0, il, seq_id);
                    }
                    if (index_read == nullptr) {
                        index_read = mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id);
                    }

                    // scorer + attention per token; reads ordered after BOTH
                    // stores (attn_read/index_read are the post-store views of
                    // the second store; earlier-token invisibility is enforced
                    // by the per-row masks, so seeing the newer row is safe).
                    ggml_tensor * outs[3] = { nullptr, nullptr, nullptr };
                    for (int t = 0; t < (int) n_tokens; ++t) {
                        ggml_tensor * k_all_t = nullptr;
                        ggml_tensor * comp_mask_t = nullptr;
                        if (pair_gather) {
                            ggml_tensor * index_cache = dsv4_cache_view_3d(ctx0, index_read, nc_idx);
                            index_cache = ggml_reshape_2d(ctx0, index_cache, hparams.indexer_head_size, nc_idx);
                            ggml_tensor * scores_t = dsv4_build_indexer_scores_decode(ctx0,
                                    cur_c[t], qr_c[t], index_cache,
                                    layer.indexer_attn_q_b,
                                    layer.indexer_proj,
                                    pos_c[t],
                                    hparams.indexer_n_head,
                                    hparams.indexer_head_size,
                                    nc_idx, n_rot, rope_type, rope_cfg);
                            ggml_tensor * bmask_t = ggml_view_2d(ctx0, bucket_mask, nc_idx, 1,
                                    bucket_mask->nb[1], (size_t) t * bucket_mask->nb[1]);
                            scores_t = ggml_add(ctx0, scores_t, bmask_t);
                            ggml_tensor * topk = ggml_argsort_top_k(ctx0, scores_t, top_k);
                            topk = ggml_reshape_1d(ctx0, topk, top_k);
                            ggml_tensor * comp_rows = ggml_view_2d(ctx0, attn_read,
                                    attn_read->ne[0], nc_idx, attn_read->nb[1], 0);
                            ggml_tensor * kv_sel = ggml_get_rows(ctx0, comp_rows, topk);
                            kv_sel = ggml_cast(ctx0, kv_sel, GGML_TYPE_F16);
                            kv_sel = ggml_reshape_3d(ctx0, kv_sel, attn_read->ne[0], 1, top_k);
                            k_all_t = ccat(k_raw_pair, kv_sel, 2, "gather-k");
                            comp_mask_t = dsv4_new_filled_2d(ctx0, top_k, 1, 0.0f);
                        } else {
                            ggml_tensor * kv_comp_cache = dsv4_cache_view_3d(ctx0, attn_read, cs_nc_fixed);
                            k_all_t = ccat(k_raw_pair, kv_comp_cache, 2, "fixed-k");
                            comp_mask_t = ggml_view_2d(ctx0, fixed_mask, cs_nc_fixed, 1,
                                    fixed_mask->nb[1], (size_t) t * fixed_mask->nb[1]);
                        }
                        ggml_tensor * mask_t = ggml_view_2d(ctx0, raw_mask_pair,
                                raw_mask_pair->ne[0], 1, raw_mask_pair->nb[1],
                                (size_t) t * raw_mask_pair->nb[1]);
                        if (mask_t->type != GGML_TYPE_F32) {
                            // get_kq_mask_swa() is F16 under flash-attn; our comp
                            // masks are F32 inputs — unify on F32, final cast below
                            mask_t = ggml_cast(ctx0, mask_t, GGML_TYPE_F32);
                        }
                        mask_t = ccat(mask_t, comp_mask_t, 0, "mask");
                        ggml_tensor * v_all_t = k_all_t;
                        if (cparams.flash_attn) {
                            dsv4_pad_fattn_width(ctx0, &k_all_t, &mask_t);
                            v_all_t = k_all_t;
                        }
                        ggml_tensor * mask_cnv_t = cparams.flash_attn
                            ? ggml_cast(ctx0, mask_t, GGML_TYPE_F16) : mask_t;
                        outs[t] = build_attn_mha(q_c[t], k_all_t, v_all_t, nullptr, mask_cnv_t,
                                layer.attn_sinks, nullptr, nullptr, kq_scale, il);
                    }
                    cs_pair_out = ccat(outs[0], outs[1], 1, "outs");
                    if (n_tokens > 2) {
                        cs_pair_out = ccat(cs_pair_out, outs[2], 1, "outs2");
                    }
                    cb(cs_pair_out, "kqv_out_pair", il);
                } else {
                // ---- legacy single-token / generic multi-token decode ----

                ggml_tensor * attn_ape_idx = n_tokens == 1
                    ? get_dsv4_inputs()->add_index(ctx0, compress_ratio, 0, "dsv4_attn_ape_idx") : nullptr;
                ggml_tensor * attn_row_idx = n_tokens == 1
                    ? get_dsv4_inputs()->add_index(ctx0, compress_ratio, compress_ratio == 4 ? compress_ratio : 0, "dsv4_attn_row_idx") : nullptr;
                dsv4_decode_compressor dec = n_tokens == 1
                    ? dsv4_build_compressor_decode(ctx0, cur,
                            prev_attn_kv_state,
                            prev_attn_sc_state,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            n_embd_head_k,
                            n_rot,
                            first_pos,
                            compress_ratio,
                            rope_type,
                            rope_cfg,
                            norm_rms_eps,
                            attn_ape_idx,
                            attn_row_idx,
                            cs_comp_pos,
                            cs_boundary)
                    : chunk_aligned
                    ? dsv4_build_compressor_chunk_batched(ctx0, cur,
                            prev_attn_kv_state,
                            prev_attn_sc_state,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            n_embd_head_k,
                            n_rot,
                            first_pos,
                            n_tokens,
                            compress_ratio,
                            rope_type,
                            rope_cfg,
                            norm_rms_eps)
                    : dsv4_build_compressor_decode_chunk(ctx0, cur,
                            prev_attn_kv_state,
                            prev_attn_sc_state,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            ubatch,
                            n_embd_head_k,
                            n_rot,
                            n_tokens,
                            compress_ratio,
                            rope_type,
                            rope_cfg,
                            norm_rms_eps);

                dsv4_store_state_segment(ctx0, gf, dec.kv_state,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0);
                dsv4_store_state_segment(ctx0, gf, dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0);

                ggml_tensor * attn_k_cache_stored = nullptr;
                if (dec.kv_comp != nullptr) {
                    dec.kv_comp = ggml_dsv4_fp8_kv_quantize(ctx0, dec.kv_comp, n_rot);
                    attn_k_cache_stored = store_attn_cache_rows(dec.kv_comp, n_comp_before, n_comp_visible - n_comp_before,
                            cs_compressor ? cs_comp_row : nullptr);
                }

                ggml_tensor * k_raw = mctx_swa->get_k(ctx0, il);
                k_raw = ggml_reshape_3d(ctx0, k_raw, n_embd_head_k, 1, k_raw->ne[2]);
                k_all = k_raw;
                v_all = k_raw;
                attn_mask = inp_attn->self_kq_mask_swa;

                if (n_comp_visible > 0) {
                    // Read the POST-STORE cache tensor when this token wrote it, so
                    // the read is ordered after the compressed-KV write (RAW edge).
                    ggml_tensor * attn_k_cache = attn_k_cache_stored
                        ? attn_k_cache_stored
                        : mctx_dsv4->get_dsv4_attn_k(ctx0, il, seq_id);
                    // Constant-shape decode: view a fixed NC_FIXED rows of the
                    // (zero-initialized, capacity n_ctx/ratio) compressed-K cache so
                    // k_all and the comp mask are depth-INVARIANT -> the decode CUDA
                    // graph replays across compression boundaries instead of
                    // recapturing every `ratio` tokens. Rows [n_comp_visible,
                    // NC_FIXED) are zero (cache cleared at alloc) and masked to -INF.
                    // Only for the ratio-4, single-token, <=top_k regime (the
                    // measured-collapse band, ctx up to ~2048); beyond that the
                    // variable top-k gather path takes over.
                    const int64_t nc_view = use_const_shape ? cs_nc_fixed : n_comp_visible;
                    ggml_tensor * kv_comp_cache = dsv4_cache_view_3d(ctx0, attn_k_cache, nc_view);
                    // V4's KV cache is F16 (forced via llama-model.cpp). CUDA's
                    // ggml_concat now supports F16 directly (ggml-cuda/concat.cu),
                    // so concat the two F16 KV segments without the F16->F32->F16
                    // round-trip. For finite/non-NaN cache values (all healthy model
                    // activations) the old round-trip was numerically a no-op
                    // (F16->F32 is exact, F32->F16 recovers the same bits), so this
                    // is bit-identical; it only differs on NaN payload canonicalization,
                    // which never occurs in valid inference. Drops 3 cast kernels per
                    // layer/token and halves the concat's memory traffic.
                    k_all = ggml_concat(ctx0, k_raw, kv_comp_cache, 2);
                    v_all = k_all;

                    ggml_tensor * comp_mask = nullptr;
                    if (compress_ratio == 4) {
                        const dsv4_state_layout index_state_layout = dsv4_make_state_layout(compress_ratio, hparams.indexer_head_size);
                        ggml_tensor * prev_index_kv_state = dsv4_view_state_segment(ctx0, prev_kv_state_all,
                                attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);
                        ggml_tensor * prev_index_sc_state = dsv4_view_state_segment(ctx0, prev_sc_state_all,
                                attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);

                        ggml_tensor * idx_ape_idx = n_tokens == 1
                            ? get_dsv4_inputs()->add_index(ctx0, compress_ratio, 0, "dsv4_idx_ape_idx") : nullptr;
                        ggml_tensor * idx_row_idx = n_tokens == 1
                            ? get_dsv4_inputs()->add_index(ctx0, compress_ratio, compress_ratio == 4 ? compress_ratio : 0, "dsv4_idx_row_idx") : nullptr;
                        dsv4_decode_compressor index_dec = n_tokens == 1
                            ? dsv4_build_compressor_decode(ctx0, cur,
                                    prev_index_kv_state,
                                    prev_index_sc_state,
                                    layer.indexer_compressor_kv,
                                    layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape,
                                    layer.indexer_compressor_norm,
                                    hparams.indexer_head_size,
                                    n_rot,
                                    first_pos,
                                    compress_ratio,
                                    rope_type,
                                    rope_cfg,
                                    norm_rms_eps,
                                    idx_ape_idx,
                                    idx_row_idx,
                                    cs_comp_pos,
                                    cs_boundary)
                            : chunk_aligned
                            ? dsv4_build_compressor_chunk_batched(ctx0, cur,
                                    prev_index_kv_state,
                                    prev_index_sc_state,
                                    layer.indexer_compressor_kv,
                                    layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape,
                                    layer.indexer_compressor_norm,
                                    hparams.indexer_head_size,
                                    n_rot,
                                    first_pos,
                                    n_tokens,
                                    compress_ratio,
                                    rope_type,
                                    rope_cfg,
                                    norm_rms_eps)
                            : dsv4_build_compressor_decode_chunk(ctx0, cur,
                                    prev_index_kv_state,
                                    prev_index_sc_state,
                                    layer.indexer_compressor_kv,
                                    layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape,
                                    layer.indexer_compressor_norm,
                                    ubatch,
                                    hparams.indexer_head_size,
                                    n_rot,
                                    n_tokens,
                                    compress_ratio,
                                    rope_type,
                                    rope_cfg,
                                    norm_rms_eps);

                        dsv4_store_state_segment(ctx0, gf, index_dec.kv_state,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                        dsv4_store_state_segment(ctx0, gf, index_dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems);

                        ggml_tensor * index_k_cache_stored = nullptr;
                        if (index_dec.kv_comp != nullptr) {
                            index_k_cache_stored = store_index_cache_rows(index_dec.kv_comp, n_comp_before, n_comp_visible - n_comp_before,
                                    cs_compressor ? cs_comp_row : nullptr);
                        }

                        if (n_tokens == 1 && n_comp_visible <= hparams.indexer_top_k) {
                            comp_mask = get_dsv4_inputs()->add_mask(ctx0,
                                    use_const_shape ? dsv4_mask_kind::COMPRESS_FIXED
                                                    : dsv4_mask_kind::COMPRESS_CAUSAL,
                                    use_const_shape ? cs_nc_fixed : n_comp_visible, n_tokens,
                                    0, n_comp_visible, 0, compress_ratio,
                                    "dsv4_attn_compress_mask");
                        } else {
                            // Gather mode buckets the indexer width so its shape
                            // changes only every DSV4_CS_BUCKET*ratio tokens; the
                            // legacy masked path uses the exact visible width
                            // (shape changes every ratio tokens -> recapture).
                            // Cap at n_comp_cache-1 to keep the scratch row out
                            // of both the score view and the gather.
                            const int64_t nc_idx = use_topk_gather
                                ? std::min<int64_t>(GGML_PAD(n_comp_visible, DSV4_CS_BUCKET), n_comp_cache - 1)
                                : n_comp_visible;
                            ggml_tensor * index_cache = dsv4_cache_view_3d(ctx0,
                                    index_k_cache_stored ? index_k_cache_stored : mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id),
                                    nc_idx);
                            index_cache = ggml_reshape_2d(ctx0, index_cache, hparams.indexer_head_size, nc_idx);
                            ggml_tensor * index_scores = n_tokens == 1
                                ? dsv4_build_indexer_scores_decode(ctx0,
                                        cur, qr, index_cache,
                                        layer.indexer_attn_q_b,
                                        layer.indexer_proj,
                                        inp_pos,
                                        hparams.indexer_n_head,
                                        hparams.indexer_head_size,
                                        nc_idx,
                                        n_rot,
                                        rope_type,
                                        rope_cfg)
                                : dsv4_build_indexer_scores_prefill(ctx0,
                                        cur, qr, dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id), n_comp_visible),
                                        layer.indexer_attn_q_b,
                                        layer.indexer_proj,
                                        inp_pos,
                                        // Prompt-chunk path: the causal indexer mask is
                                        // O(n_comp_visible x n_tokens) -> build it on GPU
                                        // (shared across layers) instead of a CPU-filled
                                        // input; note_comp_width() takes over reuse gating.
                                        !dsv4_gpu_masks_disabled()
                                            ? (get_dsv4_inputs()->note_comp_width(compress_ratio, n_comp_visible),
                                               get_gpu_comp_mask(n_comp_visible, compress_ratio))
                                            : get_dsv4_inputs()->add_mask(ctx0,
                                                dsv4_mask_kind::COMPRESS_CAUSAL,
                                                n_comp_visible, n_tokens,
                                                0, n_comp_visible, 0, compress_ratio,
                                                "dsv4_indexer_decode_causal_mask"),
                                        hparams.indexer_n_head,
                                        hparams.indexer_head_size,
                                        n_tokens,
                                        n_rot,
                                        rope_type,
                                        rope_cfg);
                            cb(index_scores, "indexer_scores", il);

                            if (use_topk_gather) {
                                // Rows [n_comp_visible, nc_idx) of the bucketed
                                // view are zero (cache zero-init) and relu-score
                                // exactly 0.0, which could outrank real rows ->
                                // mask them to -INF before the top-k (bucketed
                                // input mask, refreshed per token).
                                ggml_tensor * idx_score_mask = get_dsv4_inputs()->add_mask(ctx0,
                                        dsv4_mask_kind::COMPRESS_BUCKET,
                                        nc_idx, n_tokens,
                                        0, n_comp_cache - 1, DSV4_CS_BUCKET, compress_ratio,
                                        "dsv4_idx_score_bucket_mask");
                                index_scores = ggml_add(ctx0, index_scores, idx_score_mask);

                                const int64_t top_k = hparams.indexer_top_k;
                                ggml_tensor * topk = ggml_argsort_top_k(ctx0, index_scores, top_k);
                                cb(topk, "indexer_topk", il);
                                topk = ggml_reshape_1d(ctx0, topk, top_k);

                                // GATHER the selected compressed-K rows: the
                                // attention runs over a constant [n_raw + top_k]
                                // width at ANY depth (same attended set as the
                                // masked path; softmax is order-invariant).
                                ggml_tensor * comp_rows = ggml_view_2d(ctx0, attn_k_cache,
                                        attn_k_cache->ne[0], nc_idx, attn_k_cache->nb[1], 0);
                                ggml_tensor * kv_sel = ggml_get_rows(ctx0, comp_rows, topk); // F32 [ne0, top_k]
                                kv_sel = ggml_cast(ctx0, kv_sel, GGML_TYPE_F16);
                                kv_sel = ggml_reshape_3d(ctx0, kv_sel, attn_k_cache->ne[0], 1, top_k);
                                k_all = ggml_concat(ctx0, k_raw, kv_sel, 2);
                                v_all = k_all;

                                // Every gathered row is a valid top-k selection
                                // (n_comp_visible > top_k in this branch) -> a
                                // plain all-zero mask of constant shape.
                                comp_mask = dsv4_new_filled_2d(ctx0, top_k, n_tokens, 0.0f);
                            } else {
                                const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp_visible);
                                ggml_tensor * topk = ggml_argsort_top_k(ctx0, index_scores, top_k);
                                cb(topk, "indexer_topk", il);

                                // Sparse top-k FA (prompt chunks): the FA kernel's
                                // DSA path iterates only the listed rows and reads
                                // the mask THROUGH the indices, so the comp mask
                                // can stay plain-causal. The kernel engages only
                                // at K width >= 8192 (fattn.cu gate) - mirror it
                                // exactly, because the dense kernel would IGNORE
                                // the index list and a causal comp mask would then
                                // silently widen the attended set beyond top-k.
                                const int64_t n_raw_rows = k_raw->ne[2];
                                const int64_t fa_width   = n_raw_rows + n_comp_visible;
                                const int64_t fa_wpad    = GGML_PAD(fa_width, 256);
                                const int     sfa        = dsv4_sparse_fa_mode();
                                const int64_t n_sel      = sfa == 2 ? fa_width : n_raw_rows + top_k;
                                // list length must be %32 (the kernel's OOB tail
                                // slots read mask 0.0, not -INF); pad slots point
                                // at the first fattn pad row (mask -INF for every
                                // query) - only available when the width pads
                                const bool sel_padded_ok = n_sel % 32 == 0 || fa_wpad > fa_width;
                                const bool sparse_fa = sfa > 0 && n_tokens > 1
                                    && cparams.flash_attn
                                    && fa_wpad >= 8192
                                    && sel_padded_ok;
                                if (sparse_fa) {
                                    if (!dsv4_gpu_masks_disabled()) {
                                        get_dsv4_inputs()->note_comp_width(compress_ratio, n_comp_visible);
                                        comp_mask = get_gpu_comp_mask(n_comp_visible, compress_ratio);
                                    } else {
                                        comp_mask = get_dsv4_inputs()->add_mask(ctx0,
                                                dsv4_mask_kind::COMPRESS_CAUSAL,
                                                n_comp_visible, n_tokens,
                                                0, n_comp_visible, 0, compress_ratio,
                                                "dsv4_attn_compress_mask");
                                    }

                                    ggml_tensor * idx_f;
                                    if (sfa == 2) {
                                        // FULL list: identical attended set to the
                                        // dense path - correctness A/B, no speedup
                                        idx_f = ggml_arange(ctx0, 0.0f, (float) fa_width, 1.0f);
                                        idx_f = ggml_repeat_4d(ctx0, ggml_reshape_2d(ctx0, idx_f, fa_width, 1),
                                                fa_width, n_tokens, 1, 1);
                                    } else {
                                        // [all raw rows ++ n_raw + comp top-k]; the
                                        // raw/SWA visibility is enforced by the raw
                                        // mask read through the indices, and the
                                        // comp top-k picks are causally valid by
                                        // construction (causal mask was added to
                                        // the indexer scores before argsort)
                                        ggml_tensor * raw_f = ggml_arange(ctx0, 0.0f, (float) n_raw_rows, 1.0f);
                                        raw_f = ggml_repeat_4d(ctx0, ggml_reshape_2d(ctx0, raw_f, n_raw_rows, 1),
                                                n_raw_rows, n_tokens, 1, 1);
                                        ggml_tensor * comp_f = ggml_cast(ctx0, topk, GGML_TYPE_F32);
                                        comp_f = dsv4_add_scalar(ctx0, comp_f, (float) n_raw_rows);
                                        idx_f = ggml_concat(ctx0, raw_f, comp_f, 0);
                                    }
                                    const int64_t n_sel_pad = GGML_PAD(idx_f->ne[0], 32);
                                    if (n_sel_pad != idx_f->ne[0]) {
                                        ggml_tensor * padf = dsv4_new_filled_2d(ctx0,
                                                n_sel_pad - idx_f->ne[0], n_tokens, (float) fa_width);
                                        idx_f = ggml_concat(ctx0, idx_f, padf, 0);
                                    }
                                    sparse_topk = ggml_cast(ctx0, idx_f, GGML_TYPE_I32);
                                    cb(sparse_topk, "dsv4_sparse_fa_topk", il);
                                } else {
                                    comp_mask = dsv4_build_compressed_mask_from_topk(ctx0, index_scores, topk);
                                }
                            }
                        }
                    } else {
                        if (n_tokens > 1 && !dsv4_gpu_masks_disabled()) {
                            // Prompt-chunk path: GPU-built causal mask (shared
                            // across layers); reuse gated by note_comp_width.
                            get_dsv4_inputs()->note_comp_width(compress_ratio, n_comp_visible);
                            comp_mask = get_gpu_comp_mask(n_comp_visible, compress_ratio);
                        } else {
                            comp_mask = get_dsv4_inputs()->add_mask(ctx0,
                                    dsv4_mask_kind::COMPRESS_CAUSAL,
                                    n_comp_visible, n_tokens,
                                    0, n_comp_visible, 0, compress_ratio,
                                    "dsv4_attn_compress_mask");
                        }
                    }

                    attn_mask = ggml_concat(ctx0, attn_mask, comp_mask, 0);
                }
                } // end legacy (non-cs_pair) decode path
            }

            if (cs_pair_out != nullptr) {
                cur = cs_pair_out;
                cb(cur, "kqv_out", il);
            } else {
            if (cparams.flash_attn && k_all == v_all) {
                // Keep the FA width a multiple of 256 so the CUDA kernel for
                // head 512 accepts it (else silent CPU fallback, see helper).
                dsv4_pad_fattn_width(ctx0, &k_all, &attn_mask);
                v_all = k_all;
            }
            ggml_tensor * attn_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, attn_mask, GGML_TYPE_F16) : attn_mask;
            cur = build_attn_mha(q, k_all, v_all, nullptr, attn_mask_cnv, layer.attn_sinks, nullptr, sparse_topk, kq_scale, il);
            cb(cur, "kqv_out", il);
            }
        }
        cur = ggml_reshape_3d(ctx0, cur, n_embd_head_v, n_head, n_tokens);
        cur = dsv4_apply_rope_tail(ctx0, cur, inp_pos,
                n_embd_head_v, n_head, n_tokens, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, true);
        cur = dsv4_grouped_out(ctx0, cur, layer.attn_wo_a, layer.attn_wo_b,
                n_embd_head_v, n_head, n_out_group, n_lora_o, n_tokens);
        cb(cur, "attn_out", il);
        inpL = dsv4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        mix = dsv4_hc_pre(ctx0, inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        cur = mix.x;
        cb(cur, "hc_ffn_pre", il);
        cb(mix.mixes, "hc_ffn_pre_mixes", il);
        cb(mix.pre, "hc_ffn_pre_weights", il);
        cb(mix.post, "hc_ffn_pre_post_weights", il);
        cb(mix.comb, "hc_ffn_pre_comb", il);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);
        ggml_tensor * selected = nullptr;
        if ((uint32_t) il < hparams.n_hash_layers && !cparams.warmup) {
            GGML_ASSERT(inp_tokens != nullptr &&
                "DeepSeek V4 hash routing requires token-id input; embedding-only / multimodal input not supported");
            selected = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens);
            cb(selected, "ffn_moe_hash_topk", il);
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected);
        cb(moe_out, "ffn_moe_out", il);
        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp,   nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
        inpL = dsv4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens);
        cb(inpL, "hc_ffn_post", il);
    }

    // ------------------------------------------------------------------
    // MTP draft branch (phase A acceptance test, env-gated: DSV4_MTP_TEST +
    // DSV4_MTP_GGUF). Teacher-forced over the whole ubatch: position j
    // drafts token j+1 from embed(tok[j]) + the base model's final HC state
    // of position j-1 (zero seed at j==0, skipped by the scorer). Attention
    // is causal SWA over the current ubatch only — no MTP KV cache needed.
    // The offline tool compares argmax(draft[j]) vs argmax(base logits[j]).
    // ------------------------------------------------------------------
    if (getenv("DSV4_MTP_TEST") != nullptr && n_tokens > 1) {
        dsv4_mtp_module & mtp = dsv4_mtp_get();
        if (mtp.ok) {
            const int64_t nt = n_tokens;
            // h-path: previous-position HC, shifted right by one column
            ggml_tensor * hc_prev = ggml_view_3d(ctx0, inpL, n_embd, n_hc, nt - 1,
                    inpL->nb[1], inpL->nb[2], 0);
            ggml_tensor * hc_seed = ggml_fill(ctx0,
                    ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, n_hc, 1), 0.0f);
            hc_prev = ggml_concat(ctx0, hc_seed, hc_prev, 2);                 // [n_embd, n_hc, nt]
            ggml_tensor * hpath = ggml_rms_norm(ctx0, hc_prev, norm_rms_eps);
            hpath = ggml_mul(ctx0, hpath, mtp.t("mtp.0.hnorm.weight"));
            hpath = ggml_mul_mat(ctx0, mtp.t("mtp.0.h_proj.weight"), hpath);  // [n_embd, n_hc, nt]
            // e-path: current-token embedding
            ggml_tensor * epath = build_norm(mtp_emb0, mtp.t("mtp.0.enorm.weight"), nullptr, LLM_NORM_RMS, -1);
            epath = ggml_mul_mat(ctx0, mtp.t("mtp.0.e_proj.weight"), epath);  // [n_embd, nt]
            epath = ggml_reshape_3d(ctx0, epath, n_embd, 1, nt);
            ggml_tensor * mtp_hc = ggml_add(ctx0, hpath, epath);              // bcast over n_hc
            cb(mtp_hc, "mtp_input_hc", -1);

            const dsv4_rope_cfg mtp_rope = dsv4_make_rope_cfg(hparams, cparams, 0);

            // attention (window MLA, same shape as a compress_ratio==0 layer)
            ggml_tensor * mtp_res = mtp_hc;
            dsv4_hc_mix mtp_mix = dsv4_hc_pre(ctx0, mtp_hc,
                    mtp.t("mtp.0.hc_attn_fn.weight"), mtp.t("mtp.0.hc_attn_scale.weight"),
                    mtp.t("mtp.0.hc_attn_base.weight"),
                    n_embd, n_hc, nt, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
            ggml_tensor * mcur = build_norm(mtp_mix.x, mtp.t("mtp.0.attn_norm.weight"), nullptr, LLM_NORM_RMS, -1);
            ggml_tensor * mqr = ggml_mul_mat(ctx0, mtp.t("mtp.0.attn_q_a.weight"), mcur);
            mqr = build_norm(mqr, mtp.t("mtp.0.attn_q_a_norm.weight"), nullptr, LLM_NORM_RMS, -1);
            ggml_tensor * mq = ggml_mul_mat(ctx0, mtp.t("mtp.0.attn_q_b.weight"), mqr);
            mq = ggml_reshape_3d(ctx0, mq, n_embd_head_k, n_head, nt);
            mq = ggml_rms_norm(ctx0, mq, norm_rms_eps);
            mq = dsv4_apply_rope_tail(ctx0, mq, inp_pos,
                    n_embd_head_k, n_head, nt, n_rot, rope_type,
                    mtp_rope.n_ctx_orig, mtp_rope.freq_base, mtp_rope.freq_scale,
                    mtp_rope.ext_factor, mtp_rope.attn_factor, mtp_rope.beta_fast, mtp_rope.beta_slow, false);
            ggml_tensor * mkv = ggml_mul_mat(ctx0, mtp.t("mtp.0.attn_kv.weight"), mcur);
            mkv = build_norm(mkv, mtp.t("mtp.0.attn_kv_a_norm.weight"), nullptr, LLM_NORM_RMS, -1);
            mkv = ggml_reshape_3d(ctx0, mkv, n_embd_head_k, 1, nt);
            mkv = dsv4_apply_rope_tail(ctx0, mkv, inp_pos,
                    n_embd_head_k, 1, nt, n_rot, rope_type,
                    mtp_rope.n_ctx_orig, mtp_rope.freq_base, mtp_rope.freq_scale,
                    mtp_rope.ext_factor, mtp_rope.attn_factor, mtp_rope.beta_fast, mtp_rope.beta_slow, false);
            mkv = ggml_dsv4_fp8_kv_quantize(ctx0, mkv, n_rot);

            ggml_tensor * mk = mkv;
            ggml_tensor * mv = mkv;
            const int64_t mtp_window = hparams.n_swa > 0 ? (int64_t) hparams.n_swa : 128;
            ggml_tensor * mmask = dsv4_gpu_mask_raw_swa(ctx0, nt, mtp_window);
            if (cparams.flash_attn) {
                dsv4_pad_fattn_width(ctx0, &mk, &mmask);
                mv = mk;
            }
            ggml_tensor * mmask_cnv = cparams.flash_attn ? ggml_cast(ctx0, mmask, GGML_TYPE_F16) : mmask;
            mcur = build_attn_mha(mq, mk, mv, nullptr, mmask_cnv,
                    mtp.t("mtp.0.attn_sinks.weight"), nullptr, nullptr, kq_scale, n_layer - 1);
            cb(mcur, "mtp_kqv_out", -1);
            mcur = ggml_reshape_3d(ctx0, mcur, n_embd_head_v, n_head, nt);
            mcur = dsv4_apply_rope_tail(ctx0, mcur, inp_pos,
                    n_embd_head_v, n_head, nt, n_rot, rope_type,
                    mtp_rope.n_ctx_orig, mtp_rope.freq_base, mtp_rope.freq_scale,
                    mtp_rope.ext_factor, mtp_rope.attn_factor, mtp_rope.beta_fast, mtp_rope.beta_slow, true);
            mcur = dsv4_grouped_out(ctx0, mcur,
                    mtp.t("mtp.0.attn_output_a.weight"), mtp.t("mtp.0.attn_output_b.weight"),
                    n_embd_head_v, n_head, n_out_group, n_lora_o, nt);
            mtp_hc = dsv4_hc_post(ctx0, mcur, mtp_res, mtp_mix.post, mtp_mix.comb, n_embd, n_hc, nt);

            // FFN (MoE-256 + shared, same gating as the base layers)
            mtp_res = mtp_hc;
            mtp_mix = dsv4_hc_pre(ctx0, mtp_hc,
                    mtp.t("mtp.0.hc_ffn_fn.weight"), mtp.t("mtp.0.hc_ffn_scale.weight"),
                    mtp.t("mtp.0.hc_ffn_base.weight"),
                    n_embd, n_hc, nt, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
            mcur = build_norm(mtp_mix.x, mtp.t("mtp.0.ffn_norm.weight"), nullptr, LLM_NORM_RMS, -1);
            ggml_tensor * mmoe = build_moe_ffn(mcur,
                    mtp.t("mtp.0.ffn_gate_inp.weight"),
                    mtp.t("mtp.0.ffn_up_exps.weight"),
                    mtp.t("mtp.0.ffn_gate_exps.weight"),
                    mtp.t("mtp.0.ffn_down_exps.weight"),
                    mtp.t("mtp.0.exp_probs_b.bias"),
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    n_layer - 1,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            ggml_tensor * mshexp = build_ffn(mcur,
                    mtp.t("mtp.0.ffn_up_shexp.weight"),   nullptr, nullptr,
                    mtp.t("mtp.0.ffn_gate_shexp.weight"), nullptr, nullptr,
                    mtp.t("mtp.0.ffn_down_shexp.weight"), nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, n_layer - 1);
            mcur = ggml_add(ctx0, mmoe, mshexp);
            mtp_hc = dsv4_hc_post(ctx0, mcur, mtp_res, mtp_mix.post, mtp_mix.comb, n_embd, n_hc, nt);

            // head: hc collapse + final norm + shared output.weight -> argmax
            mcur = dsv4_hc_head(ctx0, mtp_hc,
                    mtp.t("mtp.0.hc_head_fn.weight"), mtp.t("mtp.0.hc_head_scale.weight"),
                    mtp.t("mtp.0.hc_head_base.weight"),
                    n_embd, n_hc, nt, norm_rms_eps, hparams.hc_eps);
            mcur = build_norm(mcur, mtp.t("mtp.0.norm.weight"), nullptr, LLM_NORM_RMS, -1);
            mcur = ggml_mul_mat(ctx0, model.output, mcur);                    // [n_vocab, nt]
            ggml_tensor * mtp_draft = ggml_argmax(ctx0, mcur);                // i32 [nt]
            ggml_set_name(mtp_draft, "mtp_draft_tok");
            ggml_set_output(mtp_draft);
            ggml_build_forward_expand(gf, mtp_draft);
        }
    }

    // ------------------------------------------------------------------
    // MTP speculative decode (phase B, env-gated: DSV4_MTP_SPEC).
    // Ingest half: compute the MTP-layer KV of EVERY batch token (teacher-
    // forced, seed HC comes from the client via the mtp_seed_in input) and
    // store it into the persistent 128-row ring on the MTP device. No
    // attention here — the ring is read by the candidate half below.
    // ------------------------------------------------------------------
    ggml_tensor * mtp_spec_kv_local  = nullptr; // this batch's fp8 KV [w,1,nt]
    ggml_tensor * mtp_spec_ring_read = nullptr; // ring view ordered after write
    const bool mtp_spec = getenv("DSV4_MTP_SPEC") != nullptr && dsv4_mtp_get().ok
        && inp_tokens != nullptr;
    if (mtp_spec) {
        dsv4_mtp_module & mtp = dsv4_mtp_get();
        auto * di = get_dsv4_inputs();
        const int64_t nt = n_tokens;
        ggml_tensor * seed  = di->add_mtp_seed(ctx0, n_embd, n_hc);
        ggml_tensor * seed3 = ggml_reshape_3d(ctx0, seed, n_embd, n_hc, 1);
        ggml_tensor * hc_prev = seed3;
        if (nt > 1) {
            ggml_tensor * hshift = ggml_view_3d(ctx0, inpL, n_embd, n_hc, nt - 1,
                    inpL->nb[1], inpL->nb[2], 0);
            hc_prev = ggml_concat(ctx0, seed3, hshift, 2);
        }
        ggml_tensor * h = ggml_rms_norm(ctx0, hc_prev, norm_rms_eps);
        h = ggml_mul(ctx0, h, mtp.t("mtp.0.hnorm.weight"));
        h = ggml_mul_mat(ctx0, mtp.t("mtp.0.h_proj.weight"), h);
        ggml_tensor * e = build_norm(mtp_emb0, mtp.t("mtp.0.enorm.weight"), nullptr, LLM_NORM_RMS, -1);
        e = ggml_mul_mat(ctx0, mtp.t("mtp.0.e_proj.weight"), e);
        e = ggml_reshape_3d(ctx0, e, n_embd, 1, nt);
        ggml_tensor * ihc = ggml_add(ctx0, h, e);
        dsv4_hc_mix mix_i = dsv4_hc_pre(ctx0, ihc,
                mtp.t("mtp.0.hc_attn_fn.weight"), mtp.t("mtp.0.hc_attn_scale.weight"),
                mtp.t("mtp.0.hc_attn_base.weight"),
                n_embd, n_hc, nt, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        ggml_tensor * icur = build_norm(mix_i.x, mtp.t("mtp.0.attn_norm.weight"), nullptr, LLM_NORM_RMS, -1);
        ggml_tensor * ikv = ggml_mul_mat(ctx0, mtp.t("mtp.0.attn_kv.weight"), icur);
        ikv = build_norm(ikv, mtp.t("mtp.0.attn_kv_a_norm.weight"), nullptr, LLM_NORM_RMS, -1);
        ikv = ggml_reshape_3d(ctx0, ikv, n_embd_head_k, 1, nt);
        const dsv4_rope_cfg spec_rope = dsv4_make_rope_cfg(hparams, cparams, 0);
        ikv = dsv4_apply_rope_tail(ctx0, ikv, inp_pos,
                n_embd_head_k, 1, nt, n_rot, rope_type,
                spec_rope.n_ctx_orig, spec_rope.freq_base, spec_rope.freq_scale,
                spec_rope.ext_factor, spec_rope.attn_factor, spec_rope.beta_fast, spec_rope.beta_slow, false);
        ikv = ggml_dsv4_fp8_kv_quantize(ctx0, ikv, n_rot);
        mtp_spec_kv_local = ikv;
        if (mtp.ensure_ring(ikv->type, ikv->ne[0], n_embd, n_hc)) {
            ggml_tensor * rows = di->add_mtp_ring_rows(ctx0, nt, DSV4_MTP_RING);
            ggml_tensor * kv2d = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ikv), ikv->ne[0], nt);
            ggml_tensor * updated = ggml_set_rows(ctx0, mtp.ring_kv, kv2d, rows);
            ggml_build_forward_expand(gf, updated);
            mtp_spec_ring_read = updated;
        }
    }

    if (inp_out_ids) {
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * n_hc, n_tokens);
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_outputs);
    }

    // MTP spec: expose the final HC of the last <=2 output positions into the
    // static hout buffer (read post-decode via dsv4_mtp_spec_read_hist; the
    // client feeds the right column back as the next batch's mtp_seed_in,
    // picking column 0 after a draft reject). Static copies instead of
    // cb_eval outputs: no mid-graph observation splits / syncs.
    if (mtp_spec && dsv4_mtp_get().hout != nullptr &&
        (inp_out_ids ? n_outputs : n_tokens) >= 1) {
        dsv4_mtp_module & mtp = dsv4_mtp_get();
        const int64_t n_out_hc = inp_out_ids ? n_outputs : n_tokens;
        const int64_t nh = std::min<int64_t>(n_out_hc, 3);
        ggml_tensor * hist = ggml_view_3d(ctx0, inpL, n_embd, n_hc, nh,
                inpL->nb[1], inpL->nb[2], (n_out_hc - nh) * inpL->nb[2]);
        hist = ggml_cont(ctx0, hist);
        ggml_set_name(hist, "mtp_hc_hist");
        hist = ggml_reshape_1d(ctx0, hist, n_embd * n_hc * nh);
        ggml_tensor * hview = ggml_view_1d(ctx0, mtp.hout, n_embd * n_hc * nh, 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, hist, hview));
    }

    ggml_tensor * cur = dsv4_hc_head(ctx0, inpL,
            model.output_hc_fn, model.output_hc_scale, model.output_hc_base,
            n_embd, n_hc, inp_out_ids ? n_outputs : n_tokens,
            norm_rms_eps, hparams.hc_eps);
    cb(cur, "result_hc", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);

    // ------------------------------------------------------------------
    // MTP speculative decode (phase B), candidate half: draft BOTH verify
    // outcomes ahead of time. Candidate c extends output position
    // n_out-nc+c with token argmax(base logits at that position); its draft
    // is what the client uses next cycle (c=0 if the draft was rejected,
    // c=1 if accepted). Attention = [ring | batch-local KV | own KV] with
    // the client-consistent mtp_mask input; strict '<' keeps a rejected
    // position's stale KV invisible to the reject candidate.
    // ------------------------------------------------------------------
    // Candidate drafting runs ONLY on genuine decode ubatches. On the final
    // prompt ubatch inp_out_ids selects one output row out of many tokens
    // (n_outputs < n_tokens); building candidates there put the CUDA argmax
    // draft id through a CPU get_rows(tok_embd) whose cross-backend i32 copy
    // is unreliable in this build -> OOB embedding index. Real decode keeps the
    // lookup on-device. (The server only needs the draft from decode steps.)
    const bool mtp_cand = mtp_spec && mtp_spec_ring_read != nullptr &&
        n_tokens <= 2 && (inp_out_ids == nullptr || n_outputs == n_tokens);
    if (mtp_cand) {
        dsv4_mtp_module & mtp = dsv4_mtp_get();
        auto * di = get_dsv4_inputs();
        const int64_t nt    = n_tokens;
        const int64_t n_out = inp_out_ids ? n_outputs : n_tokens;
        const int64_t nc    = std::min<int64_t>(n_out, 3);
        const int64_t spec_window = hparams.n_swa > 0 ? (int64_t) hparams.n_swa : 128;

        ggml_tensor * lg_tail = ggml_view_2d(ctx0, cur, cur->ne[0], nc,
                cur->nb[1], (n_out - nc) * cur->nb[1]);
        ggml_tensor * cand_tok = ggml_argmax(ctx0, lg_tail);                    // i32 [nc]
        ggml_tensor * embd_src = mtp.ensure_tok_embd(model.tok_embd);
        if (embd_src == nullptr) {
            embd_src = model.tok_embd;
        }
        ggml_tensor * cand_emb = ggml_get_rows(ctx0, embd_src, cand_tok);       // [n_embd, nc]

        ggml_tensor * hc_tail = ggml_view_3d(ctx0, inpL, n_embd, n_hc, nc,
                inpL->nb[1], inpL->nb[2], (n_out - nc) * inpL->nb[2]);
        ggml_tensor * h = ggml_rms_norm(ctx0, hc_tail, norm_rms_eps);
        h = ggml_mul(ctx0, h, mtp.t("mtp.0.hnorm.weight"));
        h = ggml_mul_mat(ctx0, mtp.t("mtp.0.h_proj.weight"), h);
        ggml_tensor * e = build_norm(cand_emb, mtp.t("mtp.0.enorm.weight"), nullptr, LLM_NORM_RMS, -1);
        e = ggml_mul_mat(ctx0, mtp.t("mtp.0.e_proj.weight"), e);
        e = ggml_reshape_3d(ctx0, e, n_embd, 1, nc);
        ggml_tensor * chc = ggml_add(ctx0, h, e);                               // [n_embd, n_hc, nc]

        const dsv4_rope_cfg spec_rope = dsv4_make_rope_cfg(hparams, cparams, 0);
        ggml_tensor * cpos = di->add_mtp_cand_pos(ctx0, nc);

        ggml_tensor * cres = chc;
        dsv4_hc_mix cmix = dsv4_hc_pre(ctx0, chc,
                mtp.t("mtp.0.hc_attn_fn.weight"), mtp.t("mtp.0.hc_attn_scale.weight"),
                mtp.t("mtp.0.hc_attn_base.weight"),
                n_embd, n_hc, nc, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        ggml_tensor * ccur = build_norm(cmix.x, mtp.t("mtp.0.attn_norm.weight"), nullptr, LLM_NORM_RMS, -1);
        ggml_tensor * cqr = ggml_mul_mat(ctx0, mtp.t("mtp.0.attn_q_a.weight"), ccur);
        cqr = build_norm(cqr, mtp.t("mtp.0.attn_q_a_norm.weight"), nullptr, LLM_NORM_RMS, -1);
        ggml_tensor * cq = ggml_mul_mat(ctx0, mtp.t("mtp.0.attn_q_b.weight"), cqr);
        cq = ggml_reshape_3d(ctx0, cq, n_embd_head_k, n_head, nc);
        cq = ggml_rms_norm(ctx0, cq, norm_rms_eps);
        cq = dsv4_apply_rope_tail(ctx0, cq, cpos,
                n_embd_head_k, n_head, nc, n_rot, rope_type,
                spec_rope.n_ctx_orig, spec_rope.freq_base, spec_rope.freq_scale,
                spec_rope.ext_factor, spec_rope.attn_factor, spec_rope.beta_fast, spec_rope.beta_slow, false);
        ggml_tensor * ckv = ggml_mul_mat(ctx0, mtp.t("mtp.0.attn_kv.weight"), ccur);
        ckv = build_norm(ckv, mtp.t("mtp.0.attn_kv_a_norm.weight"), nullptr, LLM_NORM_RMS, -1);
        ckv = ggml_reshape_3d(ctx0, ckv, n_embd_head_k, 1, nc);
        ckv = dsv4_apply_rope_tail(ctx0, ckv, cpos,
                n_embd_head_k, 1, nc, n_rot, rope_type,
                spec_rope.n_ctx_orig, spec_rope.freq_base, spec_rope.freq_scale,
                spec_rope.ext_factor, spec_rope.attn_factor, spec_rope.beta_fast, spec_rope.beta_slow, false);
        ckv = ggml_dsv4_fp8_kv_quantize(ctx0, ckv, n_rot);

        ggml_tensor * ring3 = ggml_reshape_3d(ctx0, mtp_spec_ring_read,
                mtp.ring_kv->ne[0], 1, DSV4_MTP_RING);
        ggml_tensor * k_all = ggml_concat(ctx0, ring3, mtp_spec_kv_local, 2);
        k_all = ggml_concat(ctx0, k_all, ckv, 2);
        ggml_tensor * cmask = di->add_mtp_mask(ctx0, DSV4_MTP_RING, nt, nc, spec_window);
        // FA width padding to a 256 multiple — but ggml_fill may not support
        // the fp8 KV type, so pad by duplicating leading ring rows (valid bit
        // patterns) under an all--inf mask extension instead.
        if (cparams.flash_attn) {
            const int64_t wid = DSV4_MTP_RING + nt + nc;
            const int64_t pad = GGML_PAD(wid, 256) - wid;
            if (pad > 0) {
                // pad < 256 <= wid always, so the leading rows of k_all itself
                // are enough (valid fp8 bit patterns, masked to -inf below)
                ggml_tensor * padk = ggml_view_3d(ctx0, k_all,
                        k_all->ne[0], 1, pad, k_all->nb[1], k_all->nb[2], 0);
                k_all = ggml_concat(ctx0, k_all, padk, 2);
                ggml_tensor * padm = dsv4_new_filled_2d(ctx0, pad, nc, -INFINITY);
                cmask = ggml_concat(ctx0, cmask, padm, 0);
            }
        }
        ggml_tensor * cmask_cnv = cparams.flash_attn ? ggml_cast(ctx0, cmask, GGML_TYPE_F16) : cmask;
        ccur = build_attn_mha(cq, k_all, k_all, nullptr, cmask_cnv,
                mtp.t("mtp.0.attn_sinks.weight"), nullptr, nullptr, kq_scale, n_layer - 1);
        ccur = ggml_reshape_3d(ctx0, ccur, n_embd_head_v, n_head, nc);
        ccur = dsv4_apply_rope_tail(ctx0, ccur, cpos,
                n_embd_head_v, n_head, nc, n_rot, rope_type,
                spec_rope.n_ctx_orig, spec_rope.freq_base, spec_rope.freq_scale,
                spec_rope.ext_factor, spec_rope.attn_factor, spec_rope.beta_fast, spec_rope.beta_slow, true);
        ccur = dsv4_grouped_out(ctx0, ccur,
                mtp.t("mtp.0.attn_output_a.weight"), mtp.t("mtp.0.attn_output_b.weight"),
                n_embd_head_v, n_head, n_out_group, n_lora_o, nc);
        chc = dsv4_hc_post(ctx0, ccur, cres, cmix.post, cmix.comb, n_embd, n_hc, nc);

        cres = chc;
        cmix = dsv4_hc_pre(ctx0, chc,
                mtp.t("mtp.0.hc_ffn_fn.weight"), mtp.t("mtp.0.hc_ffn_scale.weight"),
                mtp.t("mtp.0.hc_ffn_base.weight"),
                n_embd, n_hc, nc, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        ccur = build_norm(cmix.x, mtp.t("mtp.0.ffn_norm.weight"), nullptr, LLM_NORM_RMS, -1);
        ggml_tensor * cmoe = build_moe_ffn(ccur,
                mtp.t("mtp.0.ffn_gate_inp.weight"),
                mtp.t("mtp.0.ffn_up_exps.weight"),
                mtp.t("mtp.0.ffn_gate_exps.weight"),
                mtp.t("mtp.0.ffn_down_exps.weight"),
                mtp.t("mtp.0.exp_probs_b.bias"),
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                n_layer - 1,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        ggml_tensor * cshexp = build_ffn(ccur,
                mtp.t("mtp.0.ffn_up_shexp.weight"),   nullptr, nullptr,
                mtp.t("mtp.0.ffn_gate_shexp.weight"), nullptr, nullptr,
                mtp.t("mtp.0.ffn_down_shexp.weight"), nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, n_layer - 1);
        ccur = ggml_add(ctx0, cmoe, cshexp);
        chc = dsv4_hc_post(ctx0, ccur, cres, cmix.post, cmix.comb, n_embd, n_hc, nc);

        ccur = dsv4_hc_head(ctx0, chc,
                mtp.t("mtp.0.hc_head_fn.weight"), mtp.t("mtp.0.hc_head_scale.weight"),
                mtp.t("mtp.0.hc_head_base.weight"),
                n_embd, n_hc, nc, norm_rms_eps, hparams.hc_eps);
        ccur = build_norm(ccur, mtp.t("mtp.0.norm.weight"), nullptr, LLM_NORM_RMS, -1);
        ccur = ggml_mul_mat(ctx0, model.output, ccur);                          // [n_vocab, nc]
        ggml_tensor * cdraft = ggml_argmax(ctx0, ccur);                         // i32 [nc]
        ggml_set_name(cdraft, "mtp_spec_draft");
        ggml_tensor * cdraft_f = ggml_cast(ctx0, cdraft, GGML_TYPE_F32);
        ggml_tensor * dview = ggml_view_1d(ctx0, mtp.dout, nc, 0);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, cdraft_f, dview));

        // ---- level-2 draft: chain off the LAST candidate (full-accept path) ----
        // input = h(out_hc of level-1, last col) + e(embed(draft1_last)); its
        // attention adds the last candidate's level-1 KV (position q2-1) + own.
        // Built only for K=2 verify triples (nt>=3): pair clients never read it.
        if (nc >= 3) {
            ggml_tensor * hc1 = ggml_view_3d(ctx0, chc, n_embd, n_hc, 1,
                    chc->nb[1], chc->nb[2], (size_t)(nc - 1) * chc->nb[2]);
            ggml_tensor * d1_last = ggml_view_1d(ctx0, cdraft, 1, (size_t)(nc - 1) * cdraft->nb[0]);
            ggml_tensor * e2 = ggml_get_rows(ctx0, embd_src, d1_last);          // [n_embd, 1]
            ggml_tensor * h2 = ggml_rms_norm(ctx0, hc1, norm_rms_eps);
            h2 = ggml_mul(ctx0, h2, mtp.t("mtp.0.hnorm.weight"));
            h2 = ggml_mul_mat(ctx0, mtp.t("mtp.0.h_proj.weight"), h2);
            e2 = build_norm(e2, mtp.t("mtp.0.enorm.weight"), nullptr, LLM_NORM_RMS, -1);
            e2 = ggml_mul_mat(ctx0, mtp.t("mtp.0.e_proj.weight"), e2);
            e2 = ggml_reshape_3d(ctx0, e2, n_embd, 1, 1);
            ggml_tensor * hc2 = ggml_add(ctx0, h2, e2);                          // [n_embd, n_hc, 1]

            ggml_tensor * cpos2 = di->add_mtp_cand_pos2(ctx0);
            ggml_tensor * res2 = hc2;
            dsv4_hc_mix mix2 = dsv4_hc_pre(ctx0, hc2,
                    mtp.t("mtp.0.hc_attn_fn.weight"), mtp.t("mtp.0.hc_attn_scale.weight"),
                    mtp.t("mtp.0.hc_attn_base.weight"),
                    n_embd, n_hc, 1, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
            ggml_tensor * c2 = build_norm(mix2.x, mtp.t("mtp.0.attn_norm.weight"), nullptr, LLM_NORM_RMS, -1);
            ggml_tensor * qr2 = ggml_mul_mat(ctx0, mtp.t("mtp.0.attn_q_a.weight"), c2);
            qr2 = build_norm(qr2, mtp.t("mtp.0.attn_q_a_norm.weight"), nullptr, LLM_NORM_RMS, -1);
            ggml_tensor * q2 = ggml_mul_mat(ctx0, mtp.t("mtp.0.attn_q_b.weight"), qr2);
            q2 = ggml_reshape_3d(ctx0, q2, n_embd_head_k, n_head, 1);
            q2 = ggml_rms_norm(ctx0, q2, norm_rms_eps);
            q2 = dsv4_apply_rope_tail(ctx0, q2, cpos2,
                    n_embd_head_k, n_head, 1, n_rot, rope_type,
                    spec_rope.n_ctx_orig, spec_rope.freq_base, spec_rope.freq_scale,
                    spec_rope.ext_factor, spec_rope.attn_factor, spec_rope.beta_fast, spec_rope.beta_slow, false);
            ggml_tensor * kv2 = ggml_mul_mat(ctx0, mtp.t("mtp.0.attn_kv.weight"), c2);
            kv2 = build_norm(kv2, mtp.t("mtp.0.attn_kv_a_norm.weight"), nullptr, LLM_NORM_RMS, -1);
            kv2 = ggml_reshape_3d(ctx0, kv2, n_embd_head_k, 1, 1);
            kv2 = dsv4_apply_rope_tail(ctx0, kv2, cpos2,
                    n_embd_head_k, 1, 1, n_rot, rope_type,
                    spec_rope.n_ctx_orig, spec_rope.freq_base, spec_rope.freq_scale,
                    spec_rope.ext_factor, spec_rope.attn_factor, spec_rope.beta_fast, spec_rope.beta_slow, false);
            kv2 = ggml_dsv4_fp8_kv_quantize(ctx0, kv2, n_rot);

            ggml_tensor * ckv1_last = ggml_view_3d(ctx0, ckv, ckv->ne[0], 1, 1,
                    ckv->nb[1], ckv->nb[2], (size_t)(nc - 1) * ckv->nb[2]);
            ggml_tensor * k2 = ggml_concat(ctx0, ring3, mtp_spec_kv_local, 2);
            k2 = ggml_concat(ctx0, k2, ckv1_last, 2);
            k2 = ggml_concat(ctx0, k2, kv2, 2);
            ggml_tensor * m2 = di->add_mtp_mask2(ctx0, DSV4_MTP_RING, nt, spec_window);
            if (cparams.flash_attn) {
                const int64_t wid2 = DSV4_MTP_RING + nt + 2;
                const int64_t pad2 = GGML_PAD(wid2, 256) - wid2;
                if (pad2 > 0) {
                    ggml_tensor * padk2 = ggml_view_3d(ctx0, k2,
                            k2->ne[0], 1, pad2, k2->nb[1], k2->nb[2], 0);
                    k2 = ggml_concat(ctx0, k2, padk2, 2);
                    m2 = ggml_concat(ctx0, m2, dsv4_new_filled_2d(ctx0, pad2, 1, -INFINITY), 0);
                }
            }
            ggml_tensor * m2c = cparams.flash_attn ? ggml_cast(ctx0, m2, GGML_TYPE_F16) : m2;
            c2 = build_attn_mha(q2, k2, k2, nullptr, m2c,
                    mtp.t("mtp.0.attn_sinks.weight"), nullptr, nullptr, kq_scale, n_layer - 1);
            c2 = ggml_reshape_3d(ctx0, c2, n_embd_head_v, n_head, 1);
            c2 = dsv4_apply_rope_tail(ctx0, c2, cpos2,
                    n_embd_head_v, n_head, 1, n_rot, rope_type,
                    spec_rope.n_ctx_orig, spec_rope.freq_base, spec_rope.freq_scale,
                    spec_rope.ext_factor, spec_rope.attn_factor, spec_rope.beta_fast, spec_rope.beta_slow, true);
            c2 = dsv4_grouped_out(ctx0, c2,
                    mtp.t("mtp.0.attn_output_a.weight"), mtp.t("mtp.0.attn_output_b.weight"),
                    n_embd_head_v, n_head, n_out_group, n_lora_o, 1);
            hc2 = dsv4_hc_post(ctx0, c2, res2, mix2.post, mix2.comb, n_embd, n_hc, 1);
            res2 = hc2;
            mix2 = dsv4_hc_pre(ctx0, hc2,
                    mtp.t("mtp.0.hc_ffn_fn.weight"), mtp.t("mtp.0.hc_ffn_scale.weight"),
                    mtp.t("mtp.0.hc_ffn_base.weight"),
                    n_embd, n_hc, 1, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
            c2 = build_norm(mix2.x, mtp.t("mtp.0.ffn_norm.weight"), nullptr, LLM_NORM_RMS, -1);
            ggml_tensor * moe2 = build_moe_ffn(c2,
                    mtp.t("mtp.0.ffn_gate_inp.weight"),
                    mtp.t("mtp.0.ffn_up_exps.weight"),
                    mtp.t("mtp.0.ffn_gate_exps.weight"),
                    mtp.t("mtp.0.ffn_down_exps.weight"),
                    mtp.t("mtp.0.exp_probs_b.bias"),
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    n_layer - 1,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            ggml_tensor * sh2 = build_ffn(c2,
                    mtp.t("mtp.0.ffn_up_shexp.weight"),   nullptr, nullptr,
                    mtp.t("mtp.0.ffn_gate_shexp.weight"), nullptr, nullptr,
                    mtp.t("mtp.0.ffn_down_shexp.weight"), nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, n_layer - 1);
            c2 = ggml_add(ctx0, moe2, sh2);
            hc2 = dsv4_hc_post(ctx0, c2, res2, mix2.post, mix2.comb, n_embd, n_hc, 1);
            c2 = dsv4_hc_head(ctx0, hc2,
                    mtp.t("mtp.0.hc_head_fn.weight"), mtp.t("mtp.0.hc_head_scale.weight"),
                    mtp.t("mtp.0.hc_head_base.weight"),
                    n_embd, n_hc, 1, norm_rms_eps, hparams.hc_eps);
            c2 = build_norm(c2, mtp.t("mtp.0.norm.weight"), nullptr, LLM_NORM_RMS, -1);
            c2 = ggml_mul_mat(ctx0, model.output, c2);
            ggml_tensor * d2 = ggml_argmax(ctx0, c2);                            // i32 [1]
            ggml_tensor * d2f = ggml_cast(ctx0, d2, GGML_TYPE_F32);
            ggml_tensor * dview2 = ggml_view_1d(ctx0, mtp.dout, 1, (size_t) nc * sizeof(float));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, d2f, dview2));
        }
    }
}

// ---------------------------------------------------------------------------
// MTP speculative-decode client interface (phase B). Plain C symbols so the
// test tool can declare them without a header. All are no-ops / false unless
// DSV4_MTP_GGUF was loaded.
// ---------------------------------------------------------------------------

extern "C" bool dsv4_mtp_spec_ready(void) {
    return dsv4_mtp_get().ok;
}

// Mark ring rows as holding the MTP KV of these absolute positions. Call
// after every successful llama_decode with the batch's positions (graph
// warmup/reserve runs never go through here, so they cannot corrupt state).
extern "C" void dsv4_mtp_spec_commit(const int32_t * pos, int32_t n) {
    dsv4_mtp_module & m = dsv4_mtp_get();
    if (m.ring_kv == nullptr || pos == nullptr) {
        return;
    }
    for (int32_t i = 0; i < n; ++i) {
        m.ring_pos_cpu[pos[i] % DSV4_MTP_RING] = pos[i];
    }
}

// Set the HC seed for the next batch's first token (= final-layer HC of the
// token right before it; take the right column of the mtp_hc_hist output).
extern "C" void dsv4_mtp_spec_set_seed(const float * hc, int64_t n) {
    dsv4_mtp_module & m = dsv4_mtp_get();
    m.hc_seed_cpu.assign(hc, hc + n);
}

// n_embd*n_hc (one hc_hist column), 0 until the first spec graph was built.
extern "C" int64_t dsv4_mtp_spec_hc_elems(void) {
    return dsv4_mtp_get().hc_elems;
}

// Read the last decode's draft tokens / hc_hist from the static buffers.
// Call AFTER llama_decode returned; n is clamped to what fits.
extern "C" bool dsv4_mtp_spec_read_draft(int32_t * out, int32_t n) {
    dsv4_mtp_module & m = dsv4_mtp_get();
    if (m.dout == nullptr || out == nullptr || n <= 0) {
        return false;
    }
    if (n > 4) n = 4;
    float tmp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    ggml_backend_tensor_get(m.dout, tmp, 0, (size_t) n * sizeof(float));
    for (int32_t i = 0; i < n; ++i) {
        out[i] = (int32_t) lroundf(tmp[i]);
    }
    return true;
}

extern "C" bool dsv4_mtp_spec_read_hist(float * out, int64_t n) {
    dsv4_mtp_module & m = dsv4_mtp_get();
    if (m.hout == nullptr || out == nullptr || n <= 0) {
        return false;
    }
    if (n > m.hc_elems * 3) n = m.hc_elems * 3;
    ggml_backend_tensor_get(m.hout, out, 0, (size_t) n * sizeof(float));
    return true;
}

// Rewind position p0 (a rejected draft) out of the caches. Legal ONLY when p0
// did not close a compression chunk (the client gates speculation so that
// (p0+1) % ratio != 0 for every compressed ratio): between boundaries the
// recurrent compressor state never advanced (writes go to a scratch row), so
// only the attention caches and the recurrent tail-pos marker need rewinding.
extern "C" bool dsv4_mtp_spec_rollback(llama_memory_t mem, int32_t seq_id, int32_t p0,
                                        int32_t last_pos) {
    auto * hyb = dynamic_cast<llama_memory_hybrid_iswa *>(mem);
    if (hyb == nullptr) {
        return false;
    }
    if (!hyb->get_mem_attn()->seq_rm(seq_id, p0, -1)) {
        return false;
    }
    // rewind the recurrent tail-pos marker (== last_pos after the batch) to p0-1
    hyb->get_mem_recr()->seq_add(seq_id, p0, last_pos + 1, (p0 - 1) - last_pos);
    return true;
}

// Shadow copies of the recurrent compressor states (r_l/s_l), one per layer on
// the SAME device (DtoD copies, ~us each). Lets the client speculate ACROSS
// chunk boundaries: save before a boundary-crossing verify, restore on reject.
// Off-boundary verifies need neither (state never advances between boundaries).
extern "C" bool dsv4_mtp_state_shadow(llama_memory_t mem, int32_t n_layer, int32_t op) {
    auto * hyb = dynamic_cast<llama_memory_hybrid_iswa *>(mem);
    if (hyb == nullptr) {
        return false;
    }
    llama_memory_recurrent * recr = hyb->get_mem_recr();
    if (recr == nullptr) {
        return false;
    }
    static std::vector<ggml_context *>          sctx;
    static std::vector<ggml_backend_buffer_t>   sbuf;
    static std::vector<ggml_tensor *> sr, ss;   // shadows
    static std::vector<ggml_tensor *> pr, ps;   // originals
    if (sr.empty()) {
        for (int32_t il = 0; il < n_layer; ++il) {
            ggml_tensor * r = il < (int32_t) recr->r_l.size() ? recr->r_l[il] : nullptr;
            ggml_tensor * s = il < (int32_t) recr->s_l.size() ? recr->s_l[il] : nullptr;
            if (r == nullptr || s == nullptr || r->buffer == nullptr) {
                pr.push_back(nullptr); ps.push_back(nullptr);
                sr.push_back(nullptr); ss.push_back(nullptr);
                continue;
            }
            ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, /*no_alloc*/ true };
            ggml_context * c = ggml_init(ip);
            ggml_tensor * rsh = ggml_dup_tensor(c, r);
            ggml_tensor * ssh = ggml_dup_tensor(c, s);
            ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(c,
                    ggml_backend_buffer_get_type(r->buffer));
            if (b == nullptr) {
                fprintf(stderr, "dsv4-mtp: state shadow alloc failed at layer %d\n", il);
                return false;
            }
            sctx.push_back(c); sbuf.push_back(b);
            pr.push_back(r); ps.push_back(s); sr.push_back(rsh); ss.push_back(ssh);
        }
        fprintf(stderr, "dsv4-mtp: state shadows ready (%d layers)\n", n_layer);
    }
    if (op == 1) {          // save
        for (size_t i = 0; i < pr.size(); ++i) {
            if (pr[i] == nullptr) continue;
            ggml_backend_tensor_copy(pr[i], sr[i]);
            ggml_backend_tensor_copy(ps[i], ss[i]);
        }
    } else if (op == 2) {   // restore
        for (size_t i = 0; i < pr.size(); ++i) {
            if (pr[i] == nullptr) continue;
            ggml_backend_tensor_copy(sr[i], pr[i]);
            ggml_backend_tensor_copy(ss[i], ps[i]);
        }
    }
    return true;
}
