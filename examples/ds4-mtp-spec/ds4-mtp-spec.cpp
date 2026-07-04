// DS4-Flash MTP phase-B end-to-end speculative-decode test.
//
// One process, two runs over the same prompt with greedy sampling:
//   1) baseline: plain nt=1 decode (DSV4_MTP_SPEC unset -> clean prod graph);
//   2) spec:     draft-1/verify-2 loop using the in-graph MTP branch
//                (DSV4_MTP_SPEC=1): every decode also emits mtp_spec_draft
//                (drafts for BOTH verify outcomes) and mtp_hc_hist.
// Greedy speculation is lossless, so the two token streams must be identical;
// any mismatch = correctness bug. Reports t/s for both runs + accept stats.
//
// Chunk-boundary gate: a rejected draft can only be rewound if it did not
// close a ratio-4/128 compression chunk. --gate 1 (default) also refuses to
// speculate when the ACCEPTED first token closes a chunk (protects against
// nt=2 boundary handling bugs); --gate 0 only excludes the draft position.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

extern "C" bool dsv4_mtp_spec_ready(void);
extern "C" void dsv4_mtp_spec_commit(const int32_t * pos, int32_t n);
extern "C" void dsv4_mtp_spec_set_seed(const float * hc, int64_t n);
extern "C" bool dsv4_mtp_spec_rollback(llama_memory_t mem, int32_t seq_id, int32_t p0);

struct cb_state {
    std::vector<int32_t> draft;   // mtp_spec_draft [nc]
    std::vector<float>   hist;    // mtp_hc_hist [n_embd*n_hc*nh]
    int64_t hist_cols = 0;        // nh
    int64_t hc_elems  = 0;        // n_embd*n_hc
};

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    cb_state * s = (cb_state *) user_data;
    const bool is_draft = strcmp(t->name, "mtp_spec_draft") == 0;
    const bool is_hist  = strcmp(t->name, "mtp_hc_hist") == 0;
    if (ask) {
        return is_draft || is_hist;
    }
    if (is_draft) {
        s->draft.resize(t->ne[0]);
        ggml_backend_tensor_get(t, s->draft.data(), 0, ggml_nbytes(t));
    } else if (is_hist) {
        s->hc_elems  = t->ne[0] * t->ne[1];
        s->hist_cols = t->ne[2];
        s->hist.resize((size_t)(s->hc_elems * s->hist_cols));
        ggml_backend_tensor_get(t, s->hist.data(), 0, ggml_nbytes(t));
    }
    return true;
}

static int argmax_row(const float * row, int n) {
    int best = 0;
    float bv = row[0];
    for (int v = 1; v < n; ++v) {
        if (row[v] > bv) { bv = row[v]; best = v; }
    }
    return best;
}

// draft position p closes a compression chunk?
static bool closes_chunk(llama_pos p) {
    return ((p + 1) % 4) == 0 || ((p + 1) % 128) == 0;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * text_path  = nullptr;
    int   n_ctx  = 8192;
    int   n_gen  = 256;
    int   gate   = 1;
    float ts[16] = {0};
    bool  has_ts = false;
    bool  spec_only = false;
    bool  verify2_only = false;   // diagnose: replay baseline pairs via nt=2, no spec
    bool  force_reject = false;   // diagnose: corrupt every draft -> 100% rejects
    bool  no_drafting  = false;   // diagnose: spec branch built but plain nt=1 loop

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "-m")     && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-f")     && i + 1 < argc) text_path  = argv[++i];
        else if (!strcmp(argv[i], "-c")     && i + 1 < argc) n_ctx      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")     && i + 1 < argc) n_gen      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gate") && i + 1 < argc) gate       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--spec-only")) spec_only = true;
        else if (!strcmp(argv[i], "--verify2-only")) verify2_only = true;
        else if (!strcmp(argv[i], "--force-reject")) force_reject = true;
        else if (!strcmp(argv[i], "--no-drafting"))  no_drafting  = true;
        else if (!strcmp(argv[i], "-ts")    && i + 1 < argc) {
            const char * p = argv[++i];
            int k = 0;
            while (p && *p && k < 16) { ts[k++] = (float) atof(p); p = strchr(p, ','); if (p) p++; }
            has_ts = true;
        }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (!model_path || !text_path) {
        fprintf(stderr, "usage: %s -m model.gguf -f prompt.txt [-n gen] [-c ctx] [--gate 0|1] [--spec-only] [-ts a,b,c,d]\n", argv[0]);
        return 1;
    }

    std::ifstream fin(text_path);
    std::stringstream ss; ss << fin.rdbuf();
    const std::string text = ss.str();

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    if (has_ts) { mp.tensor_split = ts; mp.split_mode = LLAMA_SPLIT_MODE_LAYER; }
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token eos = llama_vocab_eos(vocab);

    int n_tok = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, true, false);
    std::vector<llama_token> prompt(n_tok);
    llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), prompt.data(), n_tok, true, false);
    if (n_tok + n_gen + 8 > n_ctx) {
        prompt.resize(n_ctx - n_gen - 8);
        n_tok = (int) prompt.size();
    }
    fprintf(stderr, "prompt: %d tokens, generating %d, gate=%d\n", n_tok, n_gen, gate);

    cb_state st;
    llama_batch batch = llama_batch_init(512, 0, 1);

    auto make_ctx = [&](bool with_cb) -> llama_context * {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = n_ctx;
        cp.n_batch         = 512;
        cp.n_ubatch        = 512;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (with_cb) {
            cp.cb_eval           = eval_cb;
            cp.cb_eval_user_data = &st;
        }
        return llama_init_from_model(model, cp);
    };

    // prefill in 512-token chunks; logits on the last <=2 positions of every
    // chunk (the spec graph needs n_outputs>=1; the final chunk needs 2).
    auto prefill = [&](llama_context * ctx) -> bool {
        for (int pos = 0; pos < n_tok; pos += 512) {
            const int nt = std::min(512, n_tok - pos);
            batch.n_tokens = nt;
            for (int i = 0; i < nt; ++i) {
                batch.token[i]     = prompt[pos + i];
                batch.pos[i]       = pos + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = (i >= nt - 2);
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "prefill decode failed at %d\n", pos);
                return false;
            }
        }
        return true;
    };

    // ---------------- baseline: plain greedy nt=1 ----------------
    std::vector<llama_token> base_out;
    double base_dt = 0.0;
    if (!spec_only) {
        unsetenv("DSV4_MTP_SPEC");
        llama_context * ctx = make_ctx(false);
        if (!ctx || !prefill(ctx)) return 1;
        const float * lg = llama_get_logits(ctx);
        // logits rows: [nt-2, nt-1] -> last row is index 1 (or 0 if nt==1)
        int n_rows_last = std::min(2, n_tok);
        llama_token A = argmax_row(lg + (size_t)(n_rows_last - 1) * n_vocab, n_vocab);
        auto t0 = std::chrono::steady_clock::now();
        int pos = n_tok;
        while ((int) base_out.size() < n_gen) {
            base_out.push_back(A);
            if (A == eos) break;
            batch.n_tokens = 1;
            batch.token[0] = A; batch.pos[0] = pos;
            batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "base decode failed\n"); return 1; }
            A = argmax_row(llama_get_logits(ctx), n_vocab);
            pos++;
        }
        base_dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "baseline: %zu tokens in %.2fs = %.2f t/s\n",
                base_out.size(), base_dt, base_out.size() / base_dt);
        llama_free(ctx);
    }

    // ---------------- diagnostic: nt=2 replay of the baseline ----------------
    // No speculation, no rollback, no MTP branch (env stays unset): feed the
    // baseline tokens PAIRWISE through nt=2 decodes and check that both rows'
    // argmax reproduce the baseline. Divergence here = the nt=2 decode path
    // itself (masks/scalar inputs keyed to last_pos, numerics) disagrees with
    // sequential nt=1 — a correctness bug independent of speculation.
    if (verify2_only) {
        if (base_out.size() < 4) { fprintf(stderr, "need baseline first\n"); return 1; }
        llama_context * ctx = make_ctx(false);
        if (!ctx || !prefill(ctx)) return 1;
        int pos = n_tok;
        size_t i = 0;
        size_t n_bad = 0;
        while (i + 2 < base_out.size()) {
            batch.n_tokens = 2;
            batch.token[0] = base_out[i];     batch.pos[0] = pos;
            batch.token[1] = base_out[i + 1]; batch.pos[1] = pos + 1;
            for (int k = 0; k < 2; ++k) {
                batch.n_seq_id[k] = 1; batch.seq_id[k][0] = 0; batch.logits[k] = true;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "v2 decode failed\n"); return 1; }
            const float * L = llama_get_logits(ctx);
            const int a0 = argmax_row(L, n_vocab);
            const int a1 = argmax_row(L + (size_t) n_vocab, n_vocab);
            if (a0 != base_out[i + 1] || a1 != base_out[i + 2]) {
                if (n_bad < 10) {
                    // margin between the nt=2 winner and the baseline token:
                    // tiny -> numeric noise between execution paths, large -> real bug
                    const float m0 = L[a0] - L[base_out[i + 1]];
                    const float m1 = L[n_vocab + a1] - L[n_vocab + base_out[i + 2]];
                    fprintf(stderr, "v2 divergence at out-token %zu (pos %d): row0 %d vs base %d (margin %.4f) | row1 %d vs base %d (margin %.4f) | boundary p0=%d p1=%d\n",
                            i, pos, a0, base_out[i + 1], m0, a1, base_out[i + 2], m1,
                            (int) closes_chunk(pos), (int) closes_chunk(pos + 1));
                }
                n_bad++;
            }
            i += 2; pos += 2;
        }
        printf("VERIFY2: %zu pairs, %zu divergent\n", i / 2, n_bad);
        llama_free(ctx);
        llama_batch_free(batch);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    // ---------------- speculative: draft-1 / verify-2 ----------------
    setenv("DSV4_MTP_SPEC", "1", 1);
    if (getenv("DSV4_MTP_GGUF") == nullptr) {
        fprintf(stderr, "DSV4_MTP_GGUF not set\n");
        return 1;
    }
    std::vector<llama_token> spec_out;
    double spec_dt = 0.0;
    long n_single = 0, n_spec = 0, n_acc = 0, n_rej = 0;
    {
        llama_context * ctx = make_ctx(true);
        if (!ctx || !prefill(ctx)) return 1;
        if (!dsv4_mtp_spec_ready()) { fprintf(stderr, "mtp module not loaded\n"); return 1; }
        // commit whole prompt into the ring bookkeeping + set the seed
        {
            std::vector<int32_t> pp(n_tok);
            for (int i = 0; i < n_tok; ++i) pp[i] = i;
            dsv4_mtp_spec_commit(pp.data(), n_tok);
            if (st.hist_cols < 1) { fprintf(stderr, "no hc_hist captured\n"); return 1; }
            dsv4_mtp_spec_set_seed(st.hist.data() + (size_t)(st.hist_cols - 1) * st.hc_elems, st.hc_elems);
        }
        const float * lg = llama_get_logits(ctx);
        int n_rows_last = std::min(2, n_tok);
        llama_token A = argmax_row(lg + (size_t)(n_rows_last - 1) * n_vocab, n_vocab);
        llama_token D = st.draft.empty() ? -1 : st.draft.back();
        llama_memory_t mem = llama_get_memory(ctx);

        auto t0 = std::chrono::steady_clock::now();
        int pos = n_tok;   // position of A

        if (no_drafting) {
            // plain nt=1 loop with the spec branch built: measures pure branch cost
            while ((int) spec_out.size() < n_gen) {
                spec_out.push_back(A);
                if (A == eos) break;
                batch.n_tokens = 1;
                batch.token[0] = A; batch.pos[0] = pos;
                batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "nd decode failed\n"); return 1; }
                A = argmax_row(llama_get_logits(ctx), n_vocab);
                pos++;
                n_single++;
            }
        } else
        while ((int) spec_out.size() < n_gen) {
            if (A == eos) { spec_out.push_back(A); break; }
            const bool can_spec = D >= 0 &&
                !closes_chunk(pos + 1) &&                 // draft pos must be rewindable
                (gate == 0 || !closes_chunk(pos));        // gate 1: no boundary inside verify at all
            if (!can_spec) {
                spec_out.push_back(A);
                batch.n_tokens = 1;
                batch.token[0] = A; batch.pos[0] = pos;
                batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "spec single decode failed\n"); return 1; }
                int32_t cp = pos;
                dsv4_mtp_spec_commit(&cp, 1);
                dsv4_mtp_spec_set_seed(st.hist.data(), st.hc_elems); // nh==1
                A = argmax_row(llama_get_logits(ctx), n_vocab);
                D = st.draft.empty() ? -1 : st.draft[0];
                pos++;
                n_single++;
                continue;
            }
            // verify decode: [A @ pos, D @ pos+1]
            const llama_token D_use = force_reject ? (llama_token)((D + 1) % n_vocab) : D;
            batch.n_tokens = 2;
            batch.token[0] = A; batch.pos[0] = pos;
            batch.token[1] = D_use; batch.pos[1] = pos + 1;
            for (int i = 0; i < 2; ++i) {
                batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = true;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "spec verify decode failed\n"); return 1; }
            n_spec++;
            const float * L  = llama_get_logits(ctx);
            const llama_token B0 = argmax_row(L, n_vocab);
            if (B0 == D_use) {
                // accept: both tokens stand
                n_acc++;
                spec_out.push_back(A);
                spec_out.push_back(D_use);
                int32_t cp[2] = { pos, pos + 1 };
                dsv4_mtp_spec_commit(cp, 2);
                dsv4_mtp_spec_set_seed(st.hist.data() + (size_t) st.hc_elems, st.hc_elems); // col 1
                A = argmax_row(L + (size_t) n_vocab, n_vocab);
                D = st.draft.size() > 1 ? st.draft[1] : -1;
                pos += 2;
            } else {
                // reject: rewind pos+1 everywhere
                n_rej++;
                spec_out.push_back(A);
                if (!dsv4_mtp_spec_rollback(mem, 0, pos + 1)) {
                    fprintf(stderr, "rollback failed at pos %d\n", pos + 1);
                    return 1;
                }
                int32_t cp = pos;
                dsv4_mtp_spec_commit(&cp, 1);
                dsv4_mtp_spec_set_seed(st.hist.data(), st.hc_elems);   // col 0
                A = B0;
                D = st.draft.empty() ? -1 : st.draft[0];
                pos += 1;
            }
        }
        spec_dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if ((int) spec_out.size() > n_gen) spec_out.resize(n_gen);
        fprintf(stderr, "spec: %zu tokens in %.2fs = %.2f t/s | decodes: %ld single + %ld verify | accept %ld/%ld (%.1f%%)\n",
                spec_out.size(), spec_dt, spec_out.size() / spec_dt,
                n_single, n_spec, n_acc, n_spec ? n_acc + n_rej : 0,
                n_spec ? 100.0 * n_acc / (n_acc + n_rej) : 0.0);
        llama_free(ctx);
    }

    // ---------------- compare ----------------
    if (!spec_only) {
        size_t n_cmp = std::min(base_out.size(), spec_out.size());
        size_t diff = n_cmp;
        for (size_t i = 0; i < n_cmp; ++i) {
            if (base_out[i] != spec_out[i]) { diff = i; break; }
        }
        if (diff == n_cmp && base_out.size() == spec_out.size()) {
            printf("MATCH: sequences identical (%zu tokens)\n", n_cmp);
        } else {
            printf("MISMATCH at token %zu/%zu (base=%d spec=%d)\n", diff, n_cmp,
                   diff < base_out.size() ? base_out[diff] : -1,
                   diff < spec_out.size() ? spec_out[diff] : -1);
        }
        printf("baseline %.2f t/s | spec %.2f t/s | speedup x%.3f\n",
               base_out.size() / base_dt, spec_out.size() / spec_dt,
               (spec_out.size() / spec_dt) / (base_out.size() / base_dt));
    }
    // print generated text (spec run)
    {
        std::string out;
        char buf[256];
        for (llama_token t : spec_out) {
            int nn = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
            if (nn > 0) out.append(buf, nn);
        }
        printf("--- spec text (%zu tok) ---\n%.600s\n", spec_out.size(), out.c_str());
    }

    llama_batch_free(batch);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
