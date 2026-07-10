// DS4-Flash decode indexer trace and temporal-reuse simulator.
//
// This is an intentionally intrusive diagnostic: cb_eval copies the masked
// Lightning Indexer scores and exact Top-K list after every CSA layer, forcing
// graph splits and synchronizations.  Its timings are therefore meaningless.
// The captured values are used to evaluate temporal Top-K stability and to
// simulate partitioned score-cache refresh (TISA) without changing inference.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

struct step_trace {
    std::map<int, std::vector<float>>   scores;
    std::map<int, std::vector<int32_t>> topk;
};

struct cb_state {
    bool capture = false;
    step_trace current;
};

static int parse_layer(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return -1;
    }
    char * end = nullptr;
    const long il = strtol(name + n, &end, 10);
    return end && *end == '\0' && il >= 0 && il <= 10000 ? (int) il : -1;
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    cb_state * s = (cb_state *) user_data;
    if (!s->capture) {
        return false;
    }

    const int score_il = parse_layer(t->name, "indexer_scores_masked-");
    const int topk_il  = parse_layer(t->name, "indexer_topk-");
    const bool is_score = score_il >= 0 && t->type == GGML_TYPE_F32 && t->ne[1] == 1;
    const bool is_topk  = topk_il  >= 0 && t->type == GGML_TYPE_I32 && ggml_nelements(t) == 512;

    if (ask) {
        return is_score || is_topk;
    }
    if (is_score) {
        auto & dst = s->current.scores[score_il];
        dst.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
    } else if (is_topk) {
        auto & dst = s->current.topk[topk_il];
        dst.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(int32_t));
    }
    return true;
}

static double overlap_fraction(std::vector<int32_t> a, std::vector<int32_t> b) {
    if (a.empty() || b.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    size_t ia = 0;
    size_t ib = 0;
    size_t hit = 0;
    while (ia < a.size() && ib < b.size()) {
        if (a[ia] == b[ib]) {
            ++hit;
            ++ia;
            ++ib;
        } else if (a[ia] < b[ib]) {
            ++ia;
        } else {
            ++ib;
        }
    }
    return (double) hit / (double) std::min(a.size(), b.size());
}

static double score_mass_fraction(
        const std::vector<float> & exact_scores,
        const std::vector<int32_t> & exact_topk,
        const std::vector<int32_t> & approx_topk) {
    double denom = 0.0;
    double numer = 0.0;
    for (int32_t idx : exact_topk) {
        if (idx >= 0 && (size_t) idx < exact_scores.size() && std::isfinite(exact_scores[(size_t) idx])) {
            denom += exact_scores[(size_t) idx];
        }
    }
    for (int32_t idx : approx_topk) {
        if (idx >= 0 && (size_t) idx < exact_scores.size() && std::isfinite(exact_scores[(size_t) idx])) {
            numer += exact_scores[(size_t) idx];
        }
    }
    return denom > 0.0 ? numer / denom : std::numeric_limits<double>::quiet_NaN();
}

static std::vector<int32_t> cpu_topk(const std::vector<float> & scores, size_t k) {
    std::vector<int32_t> idx(scores.size());
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min(k, idx.size());
    const auto better = [&scores](int32_t a, int32_t b) {
        const float av = std::isnan(scores[a]) ? -INFINITY : scores[a];
        const float bv = std::isnan(scores[b]) ? -INFINITY : scores[b];
        return av == bv ? a < b : av > bv;
    };
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), better);
    idx.resize(k);
    return idx;
}

static double mean(const std::vector<double> & v) {
    double sum = 0.0;
    size_t n = 0;
    for (double x : v) {
        if (!std::isnan(x)) {
            sum += x;
            ++n;
        }
    }
    return n ? sum / n : std::numeric_limits<double>::quiet_NaN();
}

static float minimum(const std::vector<double> & v) {
    double out = std::numeric_limits<double>::infinity();
    for (double x : v) {
        if (!std::isnan(x)) {
            out = std::min(out, x);
        }
    }
    return std::isfinite(out) ? (float) out : std::numeric_limits<float>::quiet_NaN();
}

static int greedy_argmax(const float * logits, int n_vocab) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best = i;
            best_v = logits[i];
        }
    }
    return best;
}

static std::vector<double> simulate_tisa(
        const std::vector<step_trace> & traces,
        int il,
        int partitions,
        size_t recent_rows,
        int hot_history) {
    std::vector<double> out;
    if (traces.empty()) {
        return out;
    }
    auto s0 = traces[0].scores.find(il);
    if (s0 == traces[0].scores.end()) {
        return out;
    }
    std::vector<float> cache = s0->second;
    std::vector<std::vector<int32_t>> hot;
    const auto t0 = traces[0].topk.find(il);
    if (t0 != traces[0].topk.end()) {
        hot.push_back(t0->second);
    }
    for (size_t step = 1; step < traces.size(); ++step) {
        const auto si = traces[step].scores.find(il);
        const auto ti = traces[step].topk.find(il);
        if (si == traces[step].scores.end() || ti == traces[step].topk.end()) {
            continue;
        }
        const auto & exact = si->second;
        if (cache.size() < exact.size()) {
            cache.resize(exact.size(), -INFINITY);
        } else if (cache.size() > exact.size()) {
            cache.resize(exact.size());
        }

        // Round-robin contiguous partition. Newly valid cache rows and a small
        // recency tail are always refreshed; otherwise a just-compressed token
        // could remain -INF until its partition's next turn.
        const int part = (int) ((step - 1) % (size_t) partitions);
        const size_t lo = exact.size() * (size_t) part / (size_t) partitions;
        const size_t hi = exact.size() * (size_t) (part + 1) / (size_t) partitions;
        std::copy(exact.begin() + lo, exact.begin() + hi, cache.begin() + lo);

        size_t last_valid = 0;
        bool any_valid = false;
        for (size_t i = 0; i < exact.size(); ++i) {
            if (std::isfinite(exact[i])) {
                last_valid = i;
                any_valid = true;
                if (!std::isfinite(cache[i])) {
                    cache[i] = exact[i];
                }
            }
        }
        if (any_valid) {
            const size_t tail_lo = last_valid + 1 > recent_rows ? last_valid + 1 - recent_rows : 0;
            std::copy(exact.begin() + tail_lo, exact.begin() + last_valid + 1, cache.begin() + tail_lo);
        }

        // HISA: the previous approximate winners are cheap but high-value
        // candidates. Refresh them exactly on every step in addition to the
        // rotating background partition. Runtime never needs the exact Top-K
        // here; this recursively uses only its own prior approximate lists.
        const size_t h0 = hot.size() > (size_t) hot_history ? hot.size() - (size_t) hot_history : 0;
        for (size_t h = h0; h < hot.size(); ++h) {
            for (int32_t idx : hot[h]) {
                if (idx >= 0 && (size_t) idx < exact.size()) {
                    cache[(size_t) idx] = exact[(size_t) idx];
                }
            }
        }

        auto approx = cpu_topk(cache, ti->second.size());
        out.push_back(overlap_fraction(approx, ti->second));
        hot.push_back(std::move(approx));
    }
    return out;
}

struct band_result {
    std::vector<double> overlap;
    std::vector<double> score_mass;
};

struct block_result : band_result {
    std::vector<double> scan_fraction;
};

static block_result simulate_block_hisa(
        const std::vector<step_trace> & traces,
        int il,
        int partitions,
        size_t block_size,
        size_t hot_size,
        size_t recent_rows) {
    block_result out;
    if (traces.empty()) return out;
    const auto s0 = traces[0].scores.find(il);
    const auto t0 = traces[0].topk.find(il);
    if (s0 == traces[0].scores.end() || t0 == traces[0].topk.end()) return out;

    std::vector<float> cache = s0->second;
    std::vector<int32_t> hot = cpu_topk(cache, hot_size);
    for (size_t step = 1; step < traces.size(); ++step) {
        const auto si = traces[step].scores.find(il);
        const auto ti = traces[step].topk.find(il);
        if (si == traces[step].scores.end() || ti == traces[step].topk.end()) continue;
        const auto & exact = si->second;
        cache.resize(exact.size(), -INFINITY);

        const size_t n_blocks = (exact.size() + block_size - 1) / block_size;
        std::vector<uint8_t> active(n_blocks, 0);
        const int phase = (int) ((step - 1) % (size_t) partitions);
        for (size_t b = (size_t) phase; b < n_blocks; b += (size_t) partitions) active[b] = 1;
        for (int32_t idx : hot) {
            if (idx >= 0 && (size_t) idx < exact.size()) active[(size_t) idx / block_size] = 1;
        }

        size_t last_valid = 0;
        bool any_valid = false;
        for (size_t i = 0; i < exact.size(); ++i) {
            if (std::isfinite(exact[i])) {
                last_valid = i;
                any_valid = true;
                if (!std::isfinite(cache[i])) cache[i] = exact[i];
            }
        }
        if (any_valid) {
            const size_t tail_lo = last_valid + 1 > recent_rows ? last_valid + 1 - recent_rows : 0;
            for (size_t i = tail_lo; i <= last_valid; ++i) active[i / block_size] = 1;
        }

        size_t active_blocks = 0;
        for (size_t b = 0; b < n_blocks; ++b) {
            if (!active[b]) continue;
            ++active_blocks;
            const size_t lo = b * block_size;
            const size_t hi = std::min(lo + block_size, exact.size());
            std::copy(exact.begin() + lo, exact.begin() + hi, cache.begin() + lo);
        }

        const auto approx = cpu_topk(cache, ti->second.size());
        out.overlap.push_back(overlap_fraction(approx, ti->second));
        out.score_mass.push_back(score_mass_fraction(exact, ti->second, approx));
        out.scan_fraction.push_back((double) active_blocks / (double) n_blocks);
        hot = cpu_topk(cache, hot_size);
    }
    return out;
}

static band_result simulate_hisa_band(
        const std::vector<step_trace> & traces,
        int il,
        int partitions,
        size_t recent_rows,
        size_t band_size) {
    band_result out;
    if (traces.empty()) return out;
    const auto s0 = traces[0].scores.find(il);
    const auto t0 = traces[0].topk.find(il);
    if (s0 == traces[0].scores.end() || t0 == traces[0].topk.end()) return out;

    std::vector<float> cache = s0->second;
    std::vector<int32_t> hot = cpu_topk(cache, band_size);
    for (size_t step = 1; step < traces.size(); ++step) {
        const auto si = traces[step].scores.find(il);
        const auto ti = traces[step].topk.find(il);
        if (si == traces[step].scores.end() || ti == traces[step].topk.end()) continue;
        const auto & exact = si->second;
        cache.resize(exact.size(), -INFINITY);

        // A striped background refresh samples the whole history every step,
        // while the adaptive band follows rows currently close to the cutoff.
        const int part = (int) ((step - 1) % (size_t) partitions);
        for (size_t i = (size_t) part; i < exact.size(); i += (size_t) partitions) {
            cache[i] = exact[i];
        }
        for (int32_t idx : hot) {
            if (idx >= 0 && (size_t) idx < exact.size()) cache[(size_t) idx] = exact[(size_t) idx];
        }

        size_t last_valid = 0;
        bool any_valid = false;
        for (size_t i = 0; i < exact.size(); ++i) {
            if (std::isfinite(exact[i])) {
                last_valid = i;
                any_valid = true;
                if (!std::isfinite(cache[i])) cache[i] = exact[i];
            }
        }
        if (any_valid) {
            const size_t tail_lo = last_valid + 1 > recent_rows ? last_valid + 1 - recent_rows : 0;
            std::copy(exact.begin() + tail_lo, exact.begin() + last_valid + 1, cache.begin() + tail_lo);
        }

        const auto approx = cpu_topk(cache, ti->second.size());
        out.overlap.push_back(overlap_fraction(approx, ti->second));
        out.score_mass.push_back(score_mass_fraction(exact, ti->second, approx));
        hot = cpu_topk(cache, band_size);
    }
    return out;
}

static void print_analysis(const std::vector<step_trace> & traces) {
    std::map<int, std::vector<double>> temporal;
    std::vector<double> temporal_all;
    std::vector<double> cross_layer_all;

    for (size_t step = 0; step < traces.size(); ++step) {
        if (step > 0) {
            for (const auto & kv : traces[step].topk) {
                const auto prev = traces[step - 1].topk.find(kv.first);
                if (prev != traces[step - 1].topk.end()) {
                    const double x = overlap_fraction(prev->second, kv.second);
                    temporal[kv.first].push_back(x);
                    temporal_all.push_back(x);
                }
            }
        }
        const std::vector<int32_t> * prev_topk = nullptr;
        for (const auto & kv : traces[step].topk) {
            // CSA and HCA alternate, so adjacent entries in this map are the
            // neighboring CSA layers even though their layer ids differ by 2.
            if (prev_topk) {
                cross_layer_all.push_back(overlap_fraction(*prev_topk, kv.second));
            }
            prev_topk = &kv.second;
        }
    }

    printf("\nSUMMARY traces=%zu temporal_mean=%.6f temporal_min=%.6f cross_layer_mean=%.6f\n",
            traces.size(), mean(temporal_all), minimum(temporal_all), mean(cross_layer_all));
    printf("layer,temporal_mean,temporal_min,tisa_k2_mean,tisa_k2_min,tisa_k4_mean,tisa_k4_min,tisa_k8_mean,tisa_k8_min\n");

    std::vector<double> k1_sanity;
    std::vector<double> tisa_all[3];
    std::vector<double> hot_all[5];
    std::vector<double> band_all[6];
    std::vector<double> band_mass_all[6];
    std::vector<double> safe_all[5];
    std::vector<double> safe_mass_all[5];
    std::vector<double> block_all[6];
    std::vector<double> block_mass_all[6];
    std::vector<double> block_scan_all[6];
    const int ks[3] = {2, 4, 8};
    const int hot_k[5] = {2, 2, 4, 4, 8};
    const int hot_h[5] = {1, 2, 1, 2, 2};
    const int band_k[6] = {2, 2, 2, 4, 4, 4};
    const int band_n[6] = {1024, 2048, 4096, 2048, 4096, 8192};
    const int safe_k[5] = {2, 2, 4, 4, 8};
    const int safe_n[5] = {4096, 8192, 8192, 12288, 16384};
    const int block_k[6] = {4, 8, 4, 8, 4, 8};
    const int block_hot[6] = {512, 512, 768, 768, 1024, 1024};
    for (const auto & kv : traces.front().topk) {
        const int il = kv.first;
        const auto k1 = simulate_tisa(traces, il, 1, 128, 0);
        k1_sanity.insert(k1_sanity.end(), k1.begin(), k1.end());
        std::vector<double> sims[3];
        for (int j = 0; j < 3; ++j) {
            sims[j] = simulate_tisa(traces, il, ks[j], 128, 0);
            tisa_all[j].insert(tisa_all[j].end(), sims[j].begin(), sims[j].end());
        }
        for (int j = 0; j < 5; ++j) {
            const auto sim = simulate_tisa(traces, il, hot_k[j], 128, hot_h[j]);
            hot_all[j].insert(hot_all[j].end(), sim.begin(), sim.end());
        }
        for (int j = 0; j < 6; ++j) {
            const auto sim = simulate_hisa_band(traces, il, band_k[j], 128, (size_t) band_n[j]);
            band_all[j].insert(band_all[j].end(), sim.overlap.begin(), sim.overlap.end());
            band_mass_all[j].insert(band_mass_all[j].end(), sim.score_mass.begin(), sim.score_mass.end());
        }
        for (int j = 0; j < 5; ++j) {
            const auto sim = simulate_hisa_band(traces, il, safe_k[j], 128, (size_t) safe_n[j]);
            safe_all[j].insert(safe_all[j].end(), sim.overlap.begin(), sim.overlap.end());
            safe_mass_all[j].insert(safe_mass_all[j].end(), sim.score_mass.begin(), sim.score_mass.end());
        }
        for (int j = 0; j < 6; ++j) {
            const auto sim = simulate_block_hisa(traces, il, block_k[j], 32, (size_t) block_hot[j], 128);
            block_all[j].insert(block_all[j].end(), sim.overlap.begin(), sim.overlap.end());
            block_mass_all[j].insert(block_mass_all[j].end(), sim.score_mass.begin(), sim.score_mass.end());
            block_scan_all[j].insert(block_scan_all[j].end(), sim.scan_fraction.begin(), sim.scan_fraction.end());
        }
        printf("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                il, mean(temporal[il]), minimum(temporal[il]),
                mean(sims[0]), minimum(sims[0]),
                mean(sims[1]), minimum(sims[1]),
                mean(sims[2]), minimum(sims[2]));
    }
    printf("GLOBAL_TISA k2_mean=%.6f k2_min=%.6f k4_mean=%.6f k4_min=%.6f k8_mean=%.6f k8_min=%.6f\n",
            mean(tisa_all[0]), minimum(tisa_all[0]),
            mean(tisa_all[1]), minimum(tisa_all[1]),
            mean(tisa_all[2]), minimum(tisa_all[2]));
    printf("SANITY full_refresh_mean=%.6f full_refresh_min=%.6f\n",
            mean(k1_sanity), minimum(k1_sanity));
    printf("GLOBAL_HISA "
           "k2_h1_mean=%.6f k2_h1_min=%.6f k2_h2_mean=%.6f k2_h2_min=%.6f "
           "k4_h1_mean=%.6f k4_h1_min=%.6f k4_h2_mean=%.6f k4_h2_min=%.6f "
           "k8_h2_mean=%.6f k8_h2_min=%.6f\n",
            mean(hot_all[0]), minimum(hot_all[0]), mean(hot_all[1]), minimum(hot_all[1]),
            mean(hot_all[2]), minimum(hot_all[2]), mean(hot_all[3]), minimum(hot_all[3]),
            mean(hot_all[4]), minimum(hot_all[4]));
    printf("GLOBAL_BAND "
           "k2_b1024_mean=%.6f k2_b1024_min=%.6f "
           "k2_b2048_mean=%.6f k2_b2048_min=%.6f "
           "k2_b4096_mean=%.6f k2_b4096_min=%.6f "
           "k4_b2048_mean=%.6f k4_b2048_min=%.6f "
           "k4_b4096_mean=%.6f k4_b4096_min=%.6f "
           "k4_b8192_mean=%.6f k4_b8192_min=%.6f\n",
            mean(band_all[0]), minimum(band_all[0]), mean(band_all[1]), minimum(band_all[1]),
            mean(band_all[2]), minimum(band_all[2]), mean(band_all[3]), minimum(band_all[3]),
            mean(band_all[4]), minimum(band_all[4]), mean(band_all[5]), minimum(band_all[5]));
    printf("GLOBAL_BAND_MASS "
           "k2_b1024_mean=%.6f k2_b1024_min=%.6f "
           "k2_b2048_mean=%.6f k2_b2048_min=%.6f "
           "k2_b4096_mean=%.6f k2_b4096_min=%.6f "
           "k4_b2048_mean=%.6f k4_b2048_min=%.6f "
           "k4_b4096_mean=%.6f k4_b4096_min=%.6f "
           "k4_b8192_mean=%.6f k4_b8192_min=%.6f\n",
            mean(band_mass_all[0]), minimum(band_mass_all[0]), mean(band_mass_all[1]), minimum(band_mass_all[1]),
            mean(band_mass_all[2]), minimum(band_mass_all[2]), mean(band_mass_all[3]), minimum(band_mass_all[3]),
            mean(band_mass_all[4]), minimum(band_mass_all[4]), mean(band_mass_all[5]), minimum(band_mass_all[5]));
    printf("GLOBAL_SAFE "
           "k2_b4096=%.6f/%.6f mass=%.6f/%.6f "
           "k2_b8192=%.6f/%.6f mass=%.6f/%.6f "
           "k4_b8192=%.6f/%.6f mass=%.6f/%.6f "
           "k4_b12288=%.6f/%.6f mass=%.6f/%.6f "
           "k8_b16384=%.6f/%.6f mass=%.6f/%.6f\n",
            mean(safe_all[0]), minimum(safe_all[0]), mean(safe_mass_all[0]), minimum(safe_mass_all[0]),
            mean(safe_all[1]), minimum(safe_all[1]), mean(safe_mass_all[1]), minimum(safe_mass_all[1]),
            mean(safe_all[2]), minimum(safe_all[2]), mean(safe_mass_all[2]), minimum(safe_mass_all[2]),
            mean(safe_all[3]), minimum(safe_all[3]), mean(safe_mass_all[3]), minimum(safe_mass_all[3]),
            mean(safe_all[4]), minimum(safe_all[4]), mean(safe_mass_all[4]), minimum(safe_mass_all[4]));
    for (int j = 0; j < 6; ++j) {
        printf("GLOBAL_BLOCK k%d_hot%d overlap=%.6f/%.6f mass=%.6f/%.6f scan=%.6f/%.6f\n",
                block_k[j], block_hot[j], mean(block_all[j]), minimum(block_all[j]),
                mean(block_mass_all[j]), minimum(block_mass_all[j]),
                mean(block_scan_all[j]), minimum(block_scan_all[j]));
    }
}

template<typename T>
static bool write_value(std::ofstream & out, const T & value) {
    out.write((const char *) &value, sizeof(value));
    return out.good();
}

template<typename T>
static bool read_value(std::ifstream & in, T & value) {
    in.read((char *) &value, sizeof(value));
    return in.good();
}

template<typename T>
static bool write_map(std::ofstream & out, const std::map<int, std::vector<T>> & values) {
    const uint32_t count = (uint32_t) values.size();
    if (!write_value(out, count)) return false;
    for (const auto & kv : values) {
        const int32_t il = kv.first;
        const uint32_t n = (uint32_t) kv.second.size();
        if (!write_value(out, il) || !write_value(out, n)) return false;
        out.write((const char *) kv.second.data(), (std::streamsize) (n * sizeof(T)));
        if (!out.good()) return false;
    }
    return true;
}

template<typename T>
static bool read_map(std::ifstream & in, std::map<int, std::vector<T>> & values) {
    uint32_t count = 0;
    if (!read_value(in, count) || count > 10000) return false;
    for (uint32_t i = 0; i < count; ++i) {
        int32_t il = -1;
        uint32_t n = 0;
        if (!read_value(in, il) || !read_value(in, n) || il < 0 || n > (1u << 26)) return false;
        auto & dst = values[il];
        dst.resize(n);
        in.read((char *) dst.data(), (std::streamsize) (n * sizeof(T)));
        if (!in.good()) return false;
    }
    return true;
}

static bool save_trace(const char * path, const std::vector<step_trace> & traces) {
    std::ofstream out(path, std::ios::binary);
    const char magic[8] = {'D','S','4','T','K','0','1','\0'};
    out.write(magic, sizeof(magic));
    const uint32_t n = (uint32_t) traces.size();
    if (!out.good() || !write_value(out, n)) return false;
    for (const auto & trace : traces) {
        if (!write_map(out, trace.scores) || !write_map(out, trace.topk)) return false;
    }
    return out.good();
}

static bool load_trace(const char * path, std::vector<step_trace> & traces) {
    std::ifstream in(path, std::ios::binary);
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    uint32_t n = 0;
    if (!in.good() || memcmp(magic, "DS4TK01", 7) != 0 || !read_value(in, n) || n > 100000) return false;
    traces.resize(n);
    for (auto & trace : traces) {
        if (!read_map(in, trace.scores) || !read_map(in, trace.topk)) return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * text_path = nullptr;
    int n_ctx = 131072;
    int n_ub = 512;
    int max_toks = 0;
    int gen_steps = 16;
    const char * trace_out = nullptr;
    const char * trace_in = nullptr;
    float ts[16] = {0};
    bool has_ts = false;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "-m")  && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-f")  && i + 1 < argc) text_path = argv[++i];
        else if (!strcmp(argv[i], "-c")  && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-ub") && i + 1 < argc) n_ub = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n")  && i + 1 < argc) max_toks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g")  && i + 1 < argc) gen_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trace-out") && i + 1 < argc) trace_out = argv[++i];
        else if (!strcmp(argv[i], "--trace-in")  && i + 1 < argc) trace_in = argv[++i];
        else if (!strcmp(argv[i], "-ts") && i + 1 < argc) {
            const char * p = argv[++i];
            int k = 0;
            while (p && *p && k < 16) {
                ts[k++] = (float) atof(p);
                p = strchr(p, ',');
                if (p) {
                    ++p;
                }
            }
            has_ts = true;
        } else {
            fprintf(stderr, "unknown arg %s\n", argv[i]);
            return 1;
        }
    }
    if (trace_in) {
        std::vector<step_trace> traces;
        if (!load_trace(trace_in, traces)) {
            fprintf(stderr, "cannot load trace %s\n", trace_in);
            return 1;
        }
        print_analysis(traces);
        return 0;
    }
    if (!model_path || !text_path || gen_steps < 2) {
        fprintf(stderr, "usage: %s -m model.gguf -f text.txt [-c n_ctx] [-ub ubatch] [-n prompt_tokens] [-g decode_steps] [-ts a,b,c,d] [--trace-out file]\n", argv[0]);
        fprintf(stderr, "       %s --trace-in file\n", argv[0]);
        return 1;
    }

    // Match the exact attention/indexer path. MoE performance flags are left
    // to the caller: they do not change the traced values and may consume the
    // small amount of VRAM needed by a full 128K diagnostic context.
    setenv("GGML_CUDA_P2P", "1", 0);
    setenv("DSV4_CONSTANT_SHAPE", "1", 0);
    setenv("DSV4_DECODE_FUSED_IDX", "1", 0);
    setenv("DSV4_SPARSE_FA", "1", 0);
    setenv("DSV4_FA_UNION", "1", 0);
    setenv("DSV4_IDX_SKIP", "1", 0);

    std::ifstream fin(text_path);
    if (!fin) {
        fprintf(stderr, "cannot read %s\n", text_path);
        return 1;
    }
    std::stringstream ss;
    ss << fin.rdbuf();
    const std::string text = ss.str();

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.use_extra_bufts = false;
    if (has_ts) {
        mp.tensor_split = ts;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    }
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }

    cb_state cb;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = std::max(4096, n_ub);
    cp.n_ubatch = n_ub;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // The raw C API default is true for compatibility, while llama-server's
    // normal default is false. A 128-token SWA model must use the compact ring
    // here; a full 128K SWA cache wastes about 5.5 GiB on this architecture.
    cp.swa_full = false;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = &cb;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "context init failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int n_tok = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, true, false);
    std::vector<llama_token> toks((size_t) n_tok);
    if (llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), n_tok, true, false) < 0) {
        fprintf(stderr, "tokenization failed\n");
        return 1;
    }
    int total = n_tok;
    if (max_toks > 0) {
        total = std::min(total, max_toks);
    }
    total = std::min(total, n_ctx - gen_steps - 1);
    if (total < 2) {
        fprintf(stderr, "prompt too short or context too small\n");
        return 1;
    }
    fprintf(stderr, "text=%d tokens using=%d context=%d ubatch=%d decode_steps=%d\n",
            n_tok, total, n_ctx, n_ub, gen_steps);

    llama_batch batch = llama_batch_init(std::max(4096, n_ub), 0, 1);
    for (int pos = 0; pos < total; pos += n_ub) {
        const int nt = std::min(n_ub, total - pos);
        batch.n_tokens = nt;
        for (int i = 0; i < nt; ++i) {
            batch.token[i] = toks[pos + i];
            batch.pos[i] = pos + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == nt - 1;
        }
        cb.capture = false;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prefill failed at pos %d\n", pos);
            return 1;
        }
        if ((pos / n_ub) % 16 == 0 || pos + nt == total) {
            fprintf(stderr, "prefill %d/%d\n", pos + nt, total);
        }
    }

    llama_token next = greedy_argmax(llama_get_logits(ctx), n_vocab);
    std::vector<step_trace> traces;
    traces.reserve((size_t) gen_steps);

    for (int step = 0; step < gen_steps; ++step) {
        const int pos = total + step;
        batch.n_tokens = 1;
        batch.token[0] = next;
        batch.pos[0] = pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        cb.current = {};
        cb.capture = true;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at step %d pos %d\n", step, pos);
            return 1;
        }
        cb.capture = false;
        traces.push_back(std::move(cb.current));
        fprintf(stderr, "trace step=%d pos=%d scores=%zu topk=%zu\n",
                step, pos, traces.back().scores.size(), traces.back().topk.size());
        next = greedy_argmax(llama_get_logits(ctx), n_vocab);
    }

    if (trace_out && !save_trace(trace_out, traces)) {
        fprintf(stderr, "cannot save trace %s\n", trace_out);
        return 1;
    }
    print_analysis(traces);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
