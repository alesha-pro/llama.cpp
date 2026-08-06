// DS4-Flash MTP phase-A offline acceptance test.
//
// Feeds real text through the base model in ubatch-sized chunks with logits
// enabled on every position. The deepseek4 graph (DSV4_MTP_TEST=1 +
// DSV4_MTP_GGUF=<mtp gguf>) grows a teacher-forced MTP draft branch whose
// argmax comes out as an i32 tensor named "mtp_draft_tok"; we grab it with
// cb_eval and compare against the base model's own greedy argmax at the same
// positions. accept = P(draft == base argmax), i.e. the speculative-decoding
// acceptance rate under greedy sampling.
//
// j==0 of every ubatch is skipped (zero HC seed). The "deep" metric only
// counts j>=128 where the MTP SWA window is fully populated.

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

struct cb_state {
    std::vector<int32_t> draft;
    bool got = false;
    std::vector<float> input_hc, main_x, logits;
    int64_t input_hc_ne[3] = {0,0,0};
    int64_t main_x_ne[2] = {0,0};
    int64_t logits_ne[2] = {0,0};
};

static void cb_grab(ggml_tensor * t, const char * name, std::vector<float> & dst, int64_t * ne, int nd) {
    if (strcmp(t->name, name) != 0 || t->type != GGML_TYPE_F32) {
        return;
    }
    dst.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
    for (int i = 0; i < nd; ++i) ne[i] = t->ne[i];
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    cb_state * s = (cb_state *) user_data;
    if (ask) {
        if (getenv("DSV4_ACCEPT_DBG") != nullptr &&
            (!strcmp(t->name, "mtp_input_hc") || !strcmp(t->name, "mtp_main_x") || !strcmp(t->name, "mtp_draft_logits"))) {
            return true;
        }
        return strcmp(t->name, "mtp_draft_tok") == 0;
    }
    if (strcmp(t->name, "mtp_draft_tok") == 0) {
        s->draft.resize(t->ne[0]);
        ggml_backend_tensor_get(t, s->draft.data(), 0, ggml_nbytes(t));
        s->got = true;
    }
    cb_grab(t, "mtp_input_hc", s->input_hc, s->input_hc_ne, 3);
    cb_grab(t, "mtp_main_x", s->main_x, s->main_x_ne, 2);
    cb_grab(t, "mtp_draft_logits", s->logits, s->logits_ne, 2);
    return true;
}

static void dump_f32(const char * path, const std::vector<float> & v, const int64_t * ne, int nd) {
    FILE * f = fopen(path, "wb");
    if (!f) return;
    int64_t n = 1; for (int i = 0; i < nd; ++i) n *= ne[i];
    fprintf(f, "nds=%d shape=", nd);
    for (int i = 0; i < nd; ++i) fprintf(f, "%lld%s", (long long) ne[i], i+1<nd?",":"\n");
    fwrite(v.data(), sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "DBG dumped %s (%ld floats)\n", path, (long) n);
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * text_path  = nullptr;
    int n_ctx    = 8192;
    int n_ub     = 512;
    int max_toks = 0; // 0 = up to n_ctx
    float ts[16] = {0};
    bool has_ts  = false;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "-m")  && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-f")  && i + 1 < argc) text_path  = argv[++i];
        else if (!strcmp(argv[i], "-c")  && i + 1 < argc) n_ctx      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ub") && i + 1 < argc) n_ub       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")  && i + 1 < argc) max_toks   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ts") && i + 1 < argc) {
            const char * p = argv[++i];
            int k = 0;
            while (p && *p && k < 16) { ts[k++] = (float) atof(p); p = strchr(p, ','); if (p) p++; }
            has_ts = true;
        }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (!model_path || !text_path) {
        fprintf(stderr, "usage: %s -m model.gguf -f text.txt [-c n_ctx] [-ub ubatch] [-n max_tokens]\n", argv[0]);
        return 1;
    }
    if (getenv("DSV4_MTP_GGUF") == nullptr) {
        fprintf(stderr, "warning: DSV4_MTP_GGUF not set — draft branch will be missing\n");
    }
    setenv("DSV4_MTP_TEST", "1", 0);

    std::ifstream fin(text_path);
    if (!fin) { fprintf(stderr, "cannot read %s\n", text_path); return 1; }
    std::stringstream ss; ss << fin.rdbuf();
    const std::string text = ss.str();

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    if (has_ts) {
        mp.tensor_split = ts;
        mp.split_mode   = LLAMA_SPLIT_MODE_LAYER;
    }
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    cb_state st;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx            = n_ctx;
    cp.n_batch          = n_ub;
    cp.n_ubatch         = n_ub;
    cp.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval          = eval_cb;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    int n_tok = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, true, false);
    std::vector<llama_token> toks(n_tok);
    llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), n_tok, true, false);
    int total = n_tok;
    if (max_toks > 0 && max_toks < total) total = max_toks;
    if (total > n_ctx) total = n_ctx;
    fprintf(stderr, "text: %d tokens (using %d), ubatch %d\n", n_tok, total, n_ub);

    llama_batch batch = llama_batch_init(n_ub, 0, 1);

    long pairs_all = 0, match_all = 0;
    long pairs_deep = 0, match_deep = 0;

    for (int pos = 0; pos < total; pos += n_ub) {
        const int nt = std::min(n_ub, total - pos);
        if (nt < 2) break;

        batch.n_tokens = nt;
        for (int i = 0; i < nt; ++i) {
            batch.token[i]     = toks[pos + i];
            batch.pos[i]       = pos + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = true;
        }

        st.got = false;
        st.draft.clear();
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at pos %d\n", pos);
            return 1;
        }
        if (!st.got || (int) st.draft.size() != nt) {
            fprintf(stderr, "no mtp_draft_tok captured at pos %d (got=%d size=%zu)\n",
                    pos, (int) st.got, st.draft.size());
            return 1;
        }

        const float * lg = llama_get_logits(ctx);
        long c_all = 0, n_all = 0, c_deep = 0, n_deep = 0;
        static const bool dbg = getenv("DSV4_ACCEPT_DBG") != nullptr;
        if (dbg && pos == 0) {
            fprintf(stderr, "DBG j | draft | base_argmax | actual_next\n");
            dump_f32("/tmp/ds4dbg-input-hc.bin", st.input_hc, st.input_hc_ne, 3);
            dump_f32("/tmp/ds4dbg-main-x.bin", st.main_x, st.main_x_ne, 2);
            dump_f32("/tmp/ds4dbg-logits.bin", st.logits, st.logits_ne, 2);
        }
        for (int j = 1; j < nt; ++j) {
            const float * row = lg + (size_t) j * n_vocab;
            int best = 0;
            float bv = row[0];
            for (int v = 1; v < n_vocab; ++v) {
                if (row[v] > bv) { bv = row[v]; best = v; }
            }
            if (dbg && pos == 0 && j <= 24) {
                fprintf(stderr, "DBG %3d | %7d | %7d | %7d | row0=%.3f row1=%.3f isnan=%d\n",
                        j, st.draft[j], best, toks[pos + j + 1 < total ? pos + j + 1 : total - 1],
                        row[0], row[1], row[0] != row[0]);
            }
            const bool hit = st.draft[j] == best;
            n_all++; c_all += hit;
            if (j >= 128) { n_deep++; c_deep += hit; }
        }
        pairs_all += n_all;  match_all += c_all;
        pairs_deep += n_deep; match_deep += c_deep;
        fprintf(stderr, "chunk @%6d nt=%3d  acc=%3ld/%3ld (%.1f%%)  deep=%3ld/%3ld\n",
                pos, nt, c_all, n_all, 100.0 * c_all / n_all, c_deep, n_deep);
    }

    printf("TOTAL pairs=%ld accept=%.4f | deep(j>=128) pairs=%ld accept=%.4f\n",
           pairs_all,  pairs_all  ? (double) match_all  / pairs_all  : 0.0,
           pairs_deep, pairs_deep ? (double) match_deep / pairs_deep : 0.0);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
