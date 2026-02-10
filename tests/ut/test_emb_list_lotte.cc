// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "nlohmann/json.hpp"
#include "utils.h"

namespace {

// ============================================================================
// Configuration: Control test behavior
// ============================================================================
constexpr int32_t MAX_DOCS_TO_LOAD = 5000;    // Hard-coded limit on documents (matches random sampling default)
constexpr int32_t MAX_QUERIES_TO_LOAD = 100;  // Hard-coded limit on queries
constexpr bool SKIP_DIRECT_TEST = false;      // Set to true to skip Direct strategy (it's slow)

// LoTTE data file paths (with official ground truth annotations)
const std::string LOTTE_DOCS_JSONL_PATH = "lotte_science_gt_docs.jsonl";
const std::string LOTTE_QUERIES_JSONL_PATH = "lotte_science_gt_queries.jsonl";

// ============================================================================
// EmbListData: Load and manage embedding list data
// ============================================================================

struct EmbListData {
    std::vector<float> vectors;
    std::vector<size_t> offsets;
    int32_t dim = 0;
    int64_t num_docs = 0;
    int64_t total_vectors = 0;

    bool
    LoadFromJsonl(const std::string& jsonl_path, int32_t max_docs) {
        std::ifstream file(jsonl_path);
        if (!file) {
            printf("Cannot open JSONL file: %s\n", jsonl_path.c_str());
            return false;
        }

        // Phase 1: Read all lines into memory
        printf("Reading JSONL file into memory...\n");
        std::vector<std::string> lines;
        lines.reserve(max_docs);
        std::string line;
        while (std::getline(file, line) && (int32_t)lines.size() < max_docs) {
            if (!line.empty()) {
                lines.push_back(std::move(line));
            }
        }
        file.close();
        printf("Read %zu lines, parsing with multiple threads...\n", lines.size());

        // Phase 2: Parse in parallel
        int num_threads = std::max(1, (int)std::thread::hardware_concurrency());
        int64_t total_lines = lines.size();
        int64_t chunk_size = (total_lines + num_threads - 1) / num_threads;

        struct ParsedDoc {
            std::vector<float> vecs;
            int32_t num_chunks = 0;
            int32_t dim = 0;
        };

        std::vector<std::vector<ParsedDoc>> thread_results(num_threads);

        auto parse_worker = [&](int tid) {
            int64_t start = tid * chunk_size;
            int64_t end = std::min(start + chunk_size, total_lines);
            if (start >= end) return;
            auto& results = thread_results[tid];
            results.reserve(end - start);

            for (int64_t i = start; i < end; ++i) {
                try {
                    auto json = nlohmann::json::parse(lines[i]);
                    const auto& chunks = json["chunks"];

                    ParsedDoc doc;
                    doc.num_chunks = chunks.size();

                    for (const auto& chunk : chunks) {
                        const auto& emb = chunk["emb"];
                        if (doc.dim == 0) {
                            doc.dim = static_cast<int32_t>(emb.size());
                        }
                        for (const auto& val : emb) {
                            doc.vecs.push_back(val.get<float>());
                        }
                    }
                    results.push_back(std::move(doc));
                } catch (const std::exception& e) {
                    printf("Error parsing line %ld: %s\n", i, e.what());
                }
            }
        };

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(parse_worker, t);
        }
        for (auto& t : threads) {
            t.join();
        }

        // Free lines to reclaim memory before merge
        { std::vector<std::string>().swap(lines); }

        // Phase 3: Merge results in order
        offsets.clear();
        offsets.push_back(0);

        size_t total_floats = 0;
        size_t total_docs_parsed = 0;
        for (const auto& tr : thread_results) {
            for (const auto& doc : tr) {
                total_floats += doc.vecs.size();
                total_docs_parsed++;
            }
        }
        vectors.reserve(total_floats);

        for (auto& tr : thread_results) {
            for (auto& doc : tr) {
                if (dim == 0) {
                    dim = doc.dim;
                }
                vectors.insert(vectors.end(), doc.vecs.begin(), doc.vecs.end());
                offsets.push_back(offsets.back() + doc.num_chunks);
                std::vector<float>().swap(doc.vecs);
            }
            std::vector<ParsedDoc>().swap(tr);
        }

        num_docs = total_docs_parsed;
        total_vectors = offsets.back();

        printf("Loaded %ld docs, %ld vectors, dim=%d from JSONL (%d threads)\n", num_docs, total_vectors, dim,
               num_threads);
        return true;
    }

    knowhere::DataSetPtr
    ToDataSet() const {
        size_t* ofs = new size_t[offsets.size()];
        std::memcpy(ofs, offsets.data(), offsets.size() * sizeof(size_t));

        auto ds = knowhere::GenDataSet(total_vectors, dim, vectors.data());
        ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(ofs));
        ds->SetIsOwner(false);
        return ds;
    }

    // Print statistics about vector counts per document
    void
    PrintStats() const {
        if (num_docs == 0)
            return;

        std::vector<int64_t> counts(num_docs);
        int64_t min_count = INT64_MAX, max_count = 0, sum_count = 0;

        for (int64_t i = 0; i < num_docs; ++i) {
            counts[i] = offsets[i + 1] - offsets[i];
            min_count = std::min(min_count, counts[i]);
            max_count = std::max(max_count, counts[i]);
            sum_count += counts[i];
        }

        std::sort(counts.begin(), counts.end());
        double avg_count = (double)sum_count / num_docs;
        int64_t median_count = counts[num_docs / 2];

        printf("Vectors per doc: min=%ld, max=%ld, avg=%.1f, median=%ld\n", min_count, max_count, avg_count,
               median_count);
    }
};

// Query data with ground truth annotations
struct QueryDataWithGT {
    std::vector<float> vectors;
    std::vector<size_t> offsets;
    std::vector<std::vector<int64_t>> gt_pids;                  // Ground truth doc IDs per query
    std::vector<std::unordered_map<int64_t, int>> gt_rels;     // doc_id -> relevance grade (graded)
    bool has_graded_relevance = false;
    int32_t dim = 0;
    int64_t num_queries = 0;
    int64_t total_vectors = 0;

    bool
    LoadFromJsonl(const std::string& jsonl_path, int32_t max_queries) {
        std::ifstream file(jsonl_path);
        if (!file) {
            printf("Cannot open JSONL file: %s\n", jsonl_path.c_str());
            return false;
        }

        std::vector<std::string> lines;
        lines.reserve(max_queries);
        std::string line;
        while (std::getline(file, line) && (int32_t)lines.size() < max_queries) {
            if (!line.empty()) {
                lines.push_back(std::move(line));
            }
        }
        file.close();

        int num_threads = std::min((int)std::thread::hardware_concurrency(), (int)lines.size());
        num_threads = std::max(1, num_threads);
        int64_t total_lines = lines.size();
        int64_t chunk_size = (total_lines + num_threads - 1) / num_threads;

        struct ParsedQuery {
            std::vector<float> vecs;
            int32_t num_chunks = 0;
            int32_t dim = 0;
            std::vector<int64_t> gt;
            std::unordered_map<int64_t, int> rels;
        };

        std::vector<std::vector<ParsedQuery>> thread_results(num_threads);

        auto parse_worker = [&](int tid) {
            int64_t start = tid * chunk_size;
            int64_t end = std::min(start + chunk_size, total_lines);
            if (start >= end) return;
            auto& results = thread_results[tid];
            results.reserve(end - start);

            for (int64_t i = start; i < end; ++i) {
                try {
                    auto json = nlohmann::json::parse(lines[i]);
                    const auto& chunks = json["chunks"];

                    ParsedQuery q;
                    q.num_chunks = chunks.size();

                    for (const auto& chunk : chunks) {
                        const auto& emb = chunk["emb"];
                        if (q.dim == 0) {
                            q.dim = static_cast<int32_t>(emb.size());
                        }
                        for (const auto& val : emb) {
                            q.vecs.push_back(val.get<float>());
                        }
                    }

                    if (json.contains("gt_pids")) {
                        for (const auto& pid : json["gt_pids"]) {
                            q.gt.push_back(pid.get<int64_t>());
                        }
                    }
                    if (json.contains("gt_rels")) {
                        for (auto& [key, val] : json["gt_rels"].items()) {
                            q.rels[std::stoll(key)] = val.get<int>();
                        }
                    } else {
                        for (auto pid : q.gt) {
                            q.rels[pid] = 1;
                        }
                    }
                    results.push_back(std::move(q));
                } catch (const std::exception& e) {
                    printf("Error parsing query line %ld: %s\n", i, e.what());
                }
            }
        };

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(parse_worker, t);
        }
        for (auto& t : threads) {
            t.join();
        }

        offsets.clear();
        offsets.push_back(0);
        gt_pids.clear();
        gt_rels.clear();

        bool found_graded = false;
        for (const auto& tr : thread_results) {
            for (const auto& q : tr) {
                if (dim == 0) {
                    dim = q.dim;
                }
                vectors.insert(vectors.end(), q.vecs.begin(), q.vecs.end());
                offsets.push_back(offsets.back() + q.num_chunks);
                gt_pids.push_back(q.gt);
                gt_rels.push_back(q.rels);
                for (const auto& [id, rel] : q.rels) {
                    if (rel > 1) found_graded = true;
                }
            }
        }
        has_graded_relevance = found_graded;

        num_queries = gt_pids.size();
        total_vectors = offsets.back();

        printf("Loaded %ld queries, %ld vectors, dim=%d from JSONL\n", num_queries, total_vectors, dim);

        int64_t total_gt = 0, min_gt = INT64_MAX, max_gt = 0;
        for (const auto& gt : gt_pids) {
            total_gt += gt.size();
            min_gt = std::min(min_gt, (int64_t)gt.size());
            max_gt = std::max(max_gt, (int64_t)gt.size());
        }
        printf("GT per query: min=%ld, max=%ld, avg=%.1f\n", min_gt, max_gt, (double)total_gt / num_queries);
        if (has_graded_relevance) {
            printf("Graded relevance: YES (nDCG will use gain=2^rel-1)\n");
        }

        return true;
    }

    knowhere::DataSetPtr
    ToDataSet() const {
        size_t* ofs = new size_t[offsets.size()];
        std::memcpy(ofs, offsets.data(), offsets.size() * sizeof(size_t));

        auto ds = knowhere::GenDataSet(total_vectors, dim, vectors.data());
        ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(ofs));
        ds->SetIsOwner(false);
        return ds;
    }

    void
    PrintStats() const {
        if (num_queries == 0)
            return;

        std::vector<int64_t> counts(num_queries);
        int64_t min_count = INT64_MAX, max_count = 0, sum_count = 0;

        for (int64_t i = 0; i < num_queries; ++i) {
            counts[i] = offsets[i + 1] - offsets[i];
            min_count = std::min(min_count, counts[i]);
            max_count = std::max(max_count, counts[i]);
            sum_count += counts[i];
        }

        std::sort(counts.begin(), counts.end());
        double avg_count = (double)sum_count / num_queries;
        int64_t median_count = counts[num_queries / 2];

        printf("Vectors per query: min=%ld, max=%ld, avg=%.1f, median=%ld\n", min_count, max_count, avg_count,
               median_count);
    }
};

}  // namespace

TEST_CASE("LoTTE ColBERT: Direct vs MUVERA", "[lotte_emb_list]") {
    // Check if LoTTE data files exist
    {
        std::ifstream docs_file(LOTTE_DOCS_JSONL_PATH);
        std::ifstream queries_file(LOTTE_QUERIES_JSONL_PATH);

        if (!docs_file.good() || !queries_file.good()) {
            printf("\n");
            printf("=============================================================\n");
            printf("LoTTE data files not found. Please prepare the data first.\n");
            printf("Expected files:\n");
            printf("  - %s\n", LOTTE_DOCS_JSONL_PATH.c_str());
            printf("  - %s\n", LOTTE_QUERIES_JSONL_PATH.c_str());
            printf("\n");
            printf("Generate LoTTE data with GT annotations:\n");
            printf("  python scripts/prepare_lotte_with_gt.py --domain science --output-dir .\n");
            printf("=============================================================\n");
            SKIP("LoTTE data files not found");
            return;
        }
    }

    // Load data
    printf("\n=== Loading LoTTE Science Data (with GT annotations) ===\n");
    EmbListData doc_data;
    QueryDataWithGT query_data;

    REQUIRE(doc_data.LoadFromJsonl(LOTTE_DOCS_JSONL_PATH, MAX_DOCS_TO_LOAD));
    doc_data.PrintStats();

    REQUIRE(query_data.LoadFromJsonl(LOTTE_QUERIES_JSONL_PATH, MAX_QUERIES_TO_LOAD));
    query_data.PrintStats();

    auto doc_ds = doc_data.ToDataSet();
    auto query_ds = query_data.ToDataSet();

    const int32_t dim = doc_data.dim;
    const int32_t num_docs = doc_data.num_docs;
    const int64_t total_vectors = doc_data.total_vectors;
    const int32_t num_queries = query_data.num_queries;



    // Multiple topk values for evaluation
    const std::vector<int32_t> topk_values = {10, 20, 50};
    const int32_t max_topk = *std::max_element(topk_values.begin(), topk_values.end());

    printf("\n=== Test Configuration ===\n");
    printf("Documents: %d, Total vectors: %ld, Dim: %d\n", num_docs, total_vectors, dim);
    printf("Queries: %d, TopK values: ", num_queries);
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf("%d%s", topk_values[i], i < topk_values.size() - 1 ? ", " : "\n");
    }
    fflush(stdout);

    // Base config - use max_topk for search, then evaluate at different k
    knowhere::Json base_conf;
    base_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
    base_conf[knowhere::meta::DIM] = dim;
    base_conf[knowhere::meta::TOPK] = max_topk;
    base_conf[knowhere::indexparam::HNSW_M] = 16;
    base_conf[knowhere::indexparam::EFCONSTRUCTION] = 200;
    base_conf[knowhere::indexparam::EF] = std::max(128, max_topk * 2);
    base_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 2.0f;

    auto version = GenTestEmbListVersionList();

    // ========== Official Ground Truth ==========
    printf("\n[Ground Truth] Using official LoTTE annotations (gt_pids)\n");
    fflush(stdout);

    // Analyze ground truth document lengths
    printf("[Ground Truth] Analyzing annotated GT document lengths...\n");
    {
        std::vector<int64_t> gt_doc_lengths;
        std::map<std::string, int> length_buckets;
        length_buckets["1-50"] = 0;
        length_buckets["51-100"] = 0;
        length_buckets["101-200"] = 0;
        length_buckets["201-500"] = 0;
        length_buckets["501-1000"] = 0;
        length_buckets["1001+"] = 0;

        for (int q = 0; q < num_queries; ++q) {
            for (int64_t doc_id : query_data.gt_pids[q]) {
                if (doc_id >= 0 && doc_id < num_docs) {
                    int64_t doc_len = doc_data.offsets[doc_id + 1] - doc_data.offsets[doc_id];
                    gt_doc_lengths.push_back(doc_len);

                    if (doc_len <= 50)
                        length_buckets["1-50"]++;
                    else if (doc_len <= 100)
                        length_buckets["51-100"]++;
                    else if (doc_len <= 200)
                        length_buckets["101-200"]++;
                    else if (doc_len <= 500)
                        length_buckets["201-500"]++;
                    else if (doc_len <= 1000)
                        length_buckets["501-1000"]++;
                    else
                        length_buckets["1001+"]++;
                }
            }
        }

        if (!gt_doc_lengths.empty()) {
            std::sort(gt_doc_lengths.begin(), gt_doc_lengths.end());
            int64_t min_len = gt_doc_lengths.front();
            int64_t max_len = gt_doc_lengths.back();
            int64_t median_len = gt_doc_lengths[gt_doc_lengths.size() / 2];
            double avg_len = 0;
            for (auto len : gt_doc_lengths) avg_len += len;
            avg_len /= gt_doc_lengths.size();

            printf("[Ground Truth] GT doc length stats: min=%ld, max=%ld, avg=%.1f, median=%ld\n", min_len, max_len,
                   avg_len, median_len);
            printf("[Ground Truth] GT doc length distribution:\n");
            int total = gt_doc_lengths.size();
            printf("    1-50:    %5d (%.1f%%)\n", length_buckets["1-50"], 100.0 * length_buckets["1-50"] / total);
            printf("   51-100:   %5d (%.1f%%)\n", length_buckets["51-100"], 100.0 * length_buckets["51-100"] / total);
            printf("  101-200:   %5d (%.1f%%)\n", length_buckets["101-200"], 100.0 * length_buckets["101-200"] / total);
            printf("  201-500:   %5d (%.1f%%)\n", length_buckets["201-500"], 100.0 * length_buckets["201-500"] / total);
            printf("  501-1000:  %5d (%.1f%%)\n", length_buckets["501-1000"],
                   100.0 * length_buckets["501-1000"] / total);
            printf("  1001+:     %5d (%.1f%%)\n", length_buckets["1001+"], 100.0 * length_buckets["1001+"] / total);
        }
        fflush(stdout);
    }

    // Recall calculation based on official GT annotations (per-query average)
    auto calc_recall_vs_gt = [&](const int64_t* result_ids, int32_t k) {
        float total_recall = 0.0f;
        int valid_queries = 0;
        for (int q = 0; q < num_queries; ++q) {
            const auto& gt = query_data.gt_pids[q];
            if (gt.empty())
                continue;

            std::unordered_set<int64_t> gt_set(gt.begin(), gt.end());
            int found = 0;
            for (int i = 0; i < k; ++i) {
                int64_t doc_id = result_ids[q * max_topk + i];
                if (doc_id >= 0 && gt_set.count(doc_id) > 0) {
                    found++;
                }
            }
            total_recall += (float)found / gt.size();
            valid_queries++;
        }
        return valid_queries > 0 ? total_recall / valid_queries : 0.0f;
    };

    // nDCG@k calculation vs GT (per-query average, supports graded relevance)
    auto calc_ndcg_vs_gt = [&](const int64_t* result_ids, int32_t k) {
        float total_ndcg = 0.0f;
        int valid_queries = 0;
        for (int q = 0; q < num_queries; ++q) {
            const auto& rels = query_data.gt_rels[q];
            if (rels.empty())
                continue;
            double dcg = 0.0;
            for (int i = 0; i < k; ++i) {
                int64_t doc_id = result_ids[q * max_topk + i];
                auto it = rels.find(doc_id);
                if (doc_id >= 0 && it != rels.end()) {
                    dcg += (std::pow(2.0, it->second) - 1.0) / std::log2(i + 2.0);
                }
            }
            std::vector<int> sorted_rels;
            sorted_rels.reserve(rels.size());
            for (const auto& [id, rel] : rels) {
                sorted_rels.push_back(rel);
            }
            std::sort(sorted_rels.rbegin(), sorted_rels.rend());
            double idcg = 0.0;
            int ideal_count = std::min((int)sorted_rels.size(), k);
            for (int i = 0; i < ideal_count; ++i) {
                idcg += (std::pow(2.0, sorted_rels[i]) - 1.0) / std::log2(i + 2.0);
            }
            total_ndcg += idcg > 0 ? (float)(dcg / idcg) : 0.0f;
            valid_queries++;
        }
        return valid_queries > 0 ? total_ndcg / valid_queries : 0.0f;
    };

    // MRR@k calculation vs GT (per-query average)
    auto calc_mrr_vs_gt = [&](const int64_t* result_ids, int32_t k) {
        float total_rr = 0.0f;
        int valid_queries = 0;
        for (int q = 0; q < num_queries; ++q) {
            const auto& gt = query_data.gt_pids[q];
            if (gt.empty())
                continue;
            std::unordered_set<int64_t> gt_set(gt.begin(), gt.end());
            for (int i = 0; i < k; ++i) {
                int64_t doc_id = result_ids[q * max_topk + i];
                if (doc_id >= 0 && gt_set.count(doc_id) > 0) {
                    total_rr += 1.0f / (i + 1);
                    break;
                }
            }
            valid_queries++;
        }
        return valid_queries > 0 ? total_rr / valid_queries : 0.0f;
    };

    // Recall calculation vs BruteForce results (per-query average)
    auto calc_recall_vs_bf = [&](const int64_t* result_ids, const int64_t* bf_ids, int32_t k) {
        if (bf_ids == nullptr)
            return 0.0f;
        float total_recall = 0.0f;
        for (int q = 0; q < num_queries; ++q) {
            std::unordered_set<int64_t> bf_set;
            int bf_count = 0;
            for (int i = 0; i < k; ++i) {
                if (bf_ids[q * max_topk + i] >= 0) {
                    bf_set.insert(bf_ids[q * max_topk + i]);
                    bf_count++;
                }
            }
            if (bf_count == 0)
                continue;

            int overlap = 0;
            for (int i = 0; i < k; ++i) {
                if (result_ids[q * max_topk + i] >= 0 && bf_set.count(result_ids[q * max_topk + i]) > 0) {
                    overlap++;
                }
            }
            total_recall += (float)overlap / bf_count;
        }
        return total_recall / num_queries;
    };

    // Calculate recalls at all topk values
    auto calc_recalls_vs_bf = [&](const int64_t* result_ids, const int64_t* bf_ids) {
        std::vector<float> recalls;
        for (int32_t k : topk_values) {
            recalls.push_back(calc_recall_vs_bf(result_ids, bf_ids, k));
        }
        return recalls;
    };

    // ========== BruteForce MaxSim ==========
    double bf_time = 0;
    std::vector<float> bf_recalls_vs_gt(topk_values.size(), 0.0f);
    const int64_t* bf_ids = nullptr;
    knowhere::DataSetPtr bf_result_ds;

    if (!SKIP_DIRECT_TEST) {
        printf("\n[BruteForce] Computing MaxSim results...\n");
        fflush(stdout);
        knowhere::Json bf_conf;
        bf_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
        bf_conf[knowhere::meta::TOPK] = max_topk;

        StopWatch sw_bf;
        auto bf_result = knowhere::BruteForce::Search<knowhere::fp32>(doc_ds, query_ds, bf_conf, nullptr);
        bf_time = sw_bf.elapsed();
        REQUIRE(bf_result.has_value());
        printf("[BruteForce] Time: %.3f s\n", bf_time);

        bf_result_ds = bf_result.value();
        bf_ids = bf_result_ds->GetIds();

        // Calculate GT metrics at each topk
        printf("[BruteForce] Recall (vs GT): ");
        for (size_t i = 0; i < topk_values.size(); ++i) {
            bf_recalls_vs_gt[i] = calc_recall_vs_gt(bf_ids, topk_values[i]);
            printf("@%d=%.1f%% ", topk_values[i], bf_recalls_vs_gt[i] * 100);
        }
        printf("\n");
        printf("[BruteForce] nDCG@10=%.4f, MRR@10=%.4f\n", calc_ndcg_vs_gt(bf_ids, 10), calc_mrr_vs_gt(bf_ids, 10));
        fflush(stdout);
    }

    // ========== Direct Strategy ==========
    double direct_build_time = 0, direct_search_time = 0;
    std::vector<float> direct_recalls(topk_values.size(), 0.0f);

    if (SKIP_DIRECT_TEST) {
        printf("\n[Direct] SKIPPED (SKIP_DIRECT_TEST = true)\n");
        fflush(stdout);
    } else {
        printf("\n[Direct] Building HNSW index for %ld vectors...\n", total_vectors);
        fflush(stdout);

        knowhere::Json direct_conf = base_conf;
        direct_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
        direct_conf["emb_list_strategy"] = "direct";

        auto direct_index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
        REQUIRE(direct_index.has_value());

        StopWatch sw_direct_build;
        auto direct_build_status = direct_index.value().Build(doc_ds, direct_conf);
        direct_build_time = sw_direct_build.elapsed();
        REQUIRE(direct_build_status == knowhere::Status::success);
        printf("[Direct] Build time: %.3f s\n", direct_build_time);

        printf("[Direct] Searching...\n");
        fflush(stdout);
        StopWatch sw_direct_search;
        auto direct_result = direct_index.value().Search(query_ds, direct_conf, nullptr);
        direct_search_time = sw_direct_search.elapsed();
        REQUIRE(direct_result.has_value());
        printf("[Direct] Search time: %.3f ms\n", direct_search_time * 1000);

        auto direct_ids = direct_result.value()->GetIds();
        direct_recalls = calc_recalls_vs_bf(direct_ids, bf_ids);
        printf("[Direct] Recall (vs BF): ");
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% ", topk_values[i], direct_recalls[i] * 100);
        }
        printf("\n");
        printf("[Direct] nDCG@10=%.4f, MRR@10=%.4f\n", calc_ndcg_vs_gt(direct_ids, 10),
               calc_mrr_vs_gt(direct_ids, 10));
        fflush(stdout);
    }

    // ========== MUVERA Strategy with Multiple Parameter Combinations ==========
    // Define parameter combinations: (num_projections, num_repeats)
    std::vector<std::pair<int32_t, int32_t>> muvera_params = {
        {2, 3}, {2, 5}, {2, 7}, {3, 3}, {3, 5}, {3, 7}, {4, 3}, {4, 5}, {4, 7}, {5, 3}, {5, 5}, {5, 7},
        {6, 3}, {6, 5}, {6, 7},
    };

    // Store results for each combination
    struct MuveraResult {
        int32_t num_projections;
        int32_t num_repeats;
        double build_time;
        double search_time;
        std::vector<float> recalls;  // recall at each topk
        float ndcg10 = 0.0f;        // nDCG@10
        float mrr10 = 0.0f;         // MRR@10
    };
    std::vector<MuveraResult> muvera_results;

    printf("\n[MUVERA] Testing %zu parameter combinations...\n", muvera_params.size());
    fflush(stdout);

    for (const auto& params : muvera_params) {
        int32_t num_proj = params.first;
        int32_t num_rep = params.second;

        printf("\n[MUVERA-%d-%d] Building HNSW index...\n", num_proj, num_rep);
        fflush(stdout);

        knowhere::Json muvera_conf = base_conf;
        muvera_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
        muvera_conf["emb_list_strategy"] = "muvera";
        muvera_conf["muvera_num_projections"] = num_proj;
        muvera_conf["muvera_num_repeats"] = num_rep;
        muvera_conf["muvera_seed"] = 42;

        auto muvera_index =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
        REQUIRE(muvera_index.has_value());

        StopWatch sw_muvera_build;
        auto muvera_build_status = muvera_index.value().Build(doc_ds, muvera_conf);
        double build_time = sw_muvera_build.elapsed();
        REQUIRE(muvera_build_status == knowhere::Status::success);
        printf("[MUVERA-%d-%d] Build time: %.3f s\n", num_proj, num_rep, build_time);

        printf("[MUVERA-%d-%d] Searching...\n", num_proj, num_rep);
        fflush(stdout);
        StopWatch sw_muvera_search;
        auto muvera_result = muvera_index.value().Search(query_ds, muvera_conf, nullptr);
        double search_time = sw_muvera_search.elapsed();
        REQUIRE(muvera_result.has_value());
        printf("[MUVERA-%d-%d] Search time: %.3f ms\n", num_proj, num_rep, search_time * 1000);

        auto muvera_ids = muvera_result.value()->GetIds();
        auto recalls = calc_recalls_vs_bf(muvera_ids, bf_ids);
        float ndcg10 = calc_ndcg_vs_gt(muvera_ids, 10);
        float mrr10 = calc_mrr_vs_gt(muvera_ids, 10);
        if (bf_ids != nullptr) {
            printf("[MUVERA-%d-%d] Recall (vs BF): ", num_proj, num_rep);
            for (size_t i = 0; i < topk_values.size(); ++i) {
                printf("@%d=%.1f%% ", topk_values[i], recalls[i] * 100);
            }
            printf("  nDCG@10=%.4f MRR@10=%.4f\n", ndcg10, mrr10);
        }
        fflush(stdout);

        muvera_results.push_back({num_proj, num_rep, build_time, search_time, recalls, ndcg10, mrr10});
    }

    // ========== Summary ==========
    printf("\n============================================================================================\n");
    printf("                              Summary (LoTTE Science)                                       \n");
    printf("============================================================================================\n");

    // Header with topk columns
    printf("| Strategy      | Build Time | Search Time |");
    for (int32_t k : topk_values) {
        printf(" R@%-3d |", k);
    }
    printf("\n");

    printf("|---------------|------------|-------------|");
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf("-------|");
    }
    printf("\n");

    if (!SKIP_DIRECT_TEST) {
        // BruteForce row
        printf("| BruteForce    | %10s | %9.2f s |", "-", bf_time);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf(" %4.1f%% |", 100.0f);
        }
        printf("\n");

        // Direct row
        printf("| Direct        | %8.2f s | %9.2f ms |", direct_build_time, direct_search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf(" %4.1f%% |", direct_recalls[i] * 100);
        }
        printf("\n");
    }

    // MUVERA rows
    for (const auto& res : muvera_results) {
        char name[32];
        snprintf(name, sizeof(name), "MUVERA-%d-%d", res.num_projections, res.num_repeats);
        printf("| %-13s | %8.2f s | %9.2f ms |", name, res.build_time, res.search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf(" %4.1f%% |", res.recalls[i] * 100);
        }
        printf("\n");
    }

    printf("============================================================================================\n");
    printf("Dataset: %d docs, %ld total vectors, avg %.1f vectors/doc\n", num_docs, total_vectors,
           (float)total_vectors / num_docs);
    if (!SKIP_DIRECT_TEST) {
        printf("BruteForce E2E (vs official GT): ");
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("R@%d=%.1f%%  ", topk_values[i], bf_recalls_vs_gt[i] * 100);
        }
        printf("nDCG@10=%.4f MRR@10=%.4f\n", calc_ndcg_vs_gt(bf_ids, 10), calc_mrr_vs_gt(bf_ids, 10));
    }
    printf("\nNote: R@K = Recall at top-K, per-query averaged, compared to BruteForce MaxSim\n");
    fflush(stdout);

    // Basic sanity checks
    if (!SKIP_DIRECT_TEST) {
        for (float r : direct_recalls) {
            REQUIRE(r >= 0.0f);
        }
    }
    for (const auto& res : muvera_results) {
        for (float r : res.recalls) {
            REQUIRE(r >= 0.0f);
        }
    }
}
