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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
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
constexpr int32_t MAX_DOCS_TO_LOAD = 10000;   // Hard-coded limit on documents
constexpr int32_t MAX_QUERIES_TO_LOAD = 500;  // Hard-coded limit on queries
constexpr bool SKIP_DIRECT_TEST = false;      // Set to true to skip Direct strategy (it's slow)

// MS MARCO data file paths (with official ground truth annotations)
const std::string MSMARCO_DOCS_JSONL_PATH = "msmarco_gt_docs.jsonl";
const std::string MSMARCO_QUERIES_JSONL_PATH = "msmarco_gt_queries.jsonl";

// LoTTE data file paths
const std::string LOTTE_DOCS_JSONL_PATH = "lotte_science_gt_docs.jsonl";
const std::string LOTTE_QUERIES_JSONL_PATH = "lotte_science_gt_queries.jsonl";

// SciFact data file paths
const std::string SCIFACT_DOCS_JSONL_PATH = "scifact_gt_docs.jsonl";
const std::string SCIFACT_QUERIES_JSONL_PATH = "scifact_gt_queries.jsonl";

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

        std::string line;
        int64_t doc_count = 0;

        offsets.clear();
        offsets.push_back(0);

        while (std::getline(file, line) && doc_count < max_docs) {
            if (line.empty())
                continue;

            try {
                auto json = nlohmann::json::parse(line);
                const auto& chunks = json["chunks"];

                for (const auto& chunk : chunks) {
                    const auto& emb = chunk["emb"];
                    if (dim == 0) {
                        dim = static_cast<int32_t>(emb.size());
                    }

                    for (const auto& val : emb) {
                        vectors.push_back(val.get<float>());
                    }
                }

                offsets.push_back(offsets.back() + chunks.size());
                doc_count++;

                if (doc_count % 1000 == 0) {
                    printf("Loaded %ld documents...\n", doc_count);
                }
            } catch (const std::exception& e) {
                printf("Error parsing line %ld: %s\n", doc_count, e.what());
                continue;
            }
        }

        file.close();

        num_docs = doc_count;
        total_vectors = offsets.back();

        printf("Loaded %ld docs, %ld vectors, dim=%d from JSONL\n", num_docs, total_vectors, dim);
        return true;
    }

    knowhere::DataSetPtr
    ToDataSet() const {
        float* data = new float[vectors.size()];
        std::memcpy(data, vectors.data(), vectors.size() * sizeof(float));

        size_t* ofs = new size_t[offsets.size()];
        std::memcpy(ofs, offsets.data(), offsets.size() * sizeof(size_t));

        auto ds = knowhere::GenDataSet(total_vectors, dim, data);
        ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(ofs));
        ds->SetIsOwner(true);
        return ds;
    }

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
    std::vector<std::vector<int64_t>> gt_pids;  // Ground truth doc IDs per query
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

        std::string line;
        int64_t query_count = 0;

        offsets.clear();
        offsets.push_back(0);
        gt_pids.clear();

        while (std::getline(file, line) && query_count < max_queries) {
            if (line.empty())
                continue;

            try {
                auto json = nlohmann::json::parse(line);
                const auto& chunks = json["chunks"];

                for (const auto& chunk : chunks) {
                    const auto& emb = chunk["emb"];
                    if (dim == 0) {
                        dim = static_cast<int32_t>(emb.size());
                    }
                    for (const auto& val : emb) {
                        vectors.push_back(val.get<float>());
                    }
                }

                offsets.push_back(offsets.back() + chunks.size());

                // Load ground truth pids
                std::vector<int64_t> query_gt;
                if (json.contains("gt_pids")) {
                    for (const auto& pid : json["gt_pids"]) {
                        query_gt.push_back(pid.get<int64_t>());
                    }
                }
                gt_pids.push_back(query_gt);

                query_count++;
            } catch (const std::exception& e) {
                printf("Error parsing line %ld: %s\n", query_count, e.what());
                continue;
            }
        }

        file.close();

        num_queries = query_count;
        total_vectors = offsets.back();

        printf("Loaded %ld queries, %ld vectors, dim=%d from JSONL\n", num_queries, total_vectors, dim);

        // Print GT stats
        int64_t total_gt = 0, min_gt = INT64_MAX, max_gt = 0;
        for (const auto& gt : gt_pids) {
            total_gt += gt.size();
            min_gt = std::min(min_gt, (int64_t)gt.size());
            max_gt = std::max(max_gt, (int64_t)gt.size());
        }
        printf("GT per query: min=%ld, max=%ld, avg=%.1f\n", min_gt, max_gt, (double)total_gt / num_queries);

        return true;
    }

    knowhere::DataSetPtr
    ToDataSet() const {
        float* data = new float[vectors.size()];
        std::memcpy(data, vectors.data(), vectors.size() * sizeof(float));

        size_t* ofs = new size_t[offsets.size()];
        std::memcpy(ofs, offsets.data(), offsets.size() * sizeof(size_t));

        auto ds = knowhere::GenDataSet(total_vectors, dim, data);
        ds->Set(knowhere::meta::EMB_LIST_OFFSET, static_cast<const size_t*>(ofs));
        ds->SetIsOwner(true);
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

TEST_CASE("MS MARCO ColBERT: Direct vs MUVERA", "[msmarco_emb_list]") {
    // Check if MS MARCO data files exist
    {
        std::ifstream docs_file(MSMARCO_DOCS_JSONL_PATH);
        std::ifstream queries_file(MSMARCO_QUERIES_JSONL_PATH);

        if (!docs_file.good() || !queries_file.good()) {
            printf("\n");
            printf("=============================================================\n");
            printf("MS MARCO data files not found. Please prepare the data first.\n");
            printf("Expected files:\n");
            printf("  - %s\n", MSMARCO_DOCS_JSONL_PATH.c_str());
            printf("  - %s\n", MSMARCO_QUERIES_JSONL_PATH.c_str());
            printf("\n");
            printf("Generate MS MARCO data with GT annotations:\n");
            printf("  python scripts/prepare_msmarco_with_gt.py --output-dir .\n");
            printf("=============================================================\n");
            SKIP("MS MARCO data files not found");
            return;
        }
    }

    // Load data
    printf("\n=== Loading MS MARCO Data (with GT annotations) ===\n");
    EmbListData doc_data;
    QueryDataWithGT query_data;

    REQUIRE(doc_data.LoadFromJsonl(MSMARCO_DOCS_JSONL_PATH, MAX_DOCS_TO_LOAD));
    doc_data.PrintStats();

    REQUIRE(query_data.LoadFromJsonl(MSMARCO_QUERIES_JSONL_PATH, MAX_QUERIES_TO_LOAD));
    query_data.PrintStats();

    auto doc_ds = doc_data.ToDataSet();
    auto query_ds = query_data.ToDataSet();

    const int32_t dim = doc_data.dim;
    const int32_t num_docs = doc_data.num_docs;
    const int64_t total_vectors = doc_data.total_vectors;
    const int32_t num_queries = std::min((int32_t)query_data.num_queries, MAX_QUERIES_TO_LOAD);

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
    base_conf[knowhere::indexparam::EFCONSTRUCTION] = 100;
    base_conf[knowhere::indexparam::EF] = 64;
    base_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 2.0f;

    auto version = GenTestEmbListVersionList();

    // ========== Official Ground Truth ==========
    printf("\n[Ground Truth] Using official MS MARCO annotations (gt_pids)\n");
    fflush(stdout);

    // E2E Recall calculation based on official GT annotations (per-query average)
    // Recall@k = (number of GT documents found in top-k results) / (total GT documents)
    // result_ids has result_k items per query
    auto calc_recall_vs_gt = [&](const int64_t* result_ids, int32_t result_k, int32_t k) {
        float total_recall = 0.0f;
        int valid_queries = 0;
        for (int q = 0; q < num_queries; ++q) {
            const auto& gt = query_data.gt_pids[q];
            if (gt.empty())
                continue;

            std::unordered_set<int64_t> gt_set(gt.begin(), gt.end());
            int found = 0;
            for (int i = 0; i < k && i < result_k; ++i) {
                int64_t doc_id = result_ids[q * result_k + i];
                if (doc_id >= 0 && gt_set.count(doc_id) > 0) {
                    found++;
                }
            }
            total_recall += (float)found / gt.size();
            valid_queries++;
        }
        return valid_queries > 0 ? total_recall / valid_queries : 0.0f;
    };

    // Recall calculation vs BruteForce results (per-query average)
    // result_ids has result_k items per query, bf_ids has bf_k items per query
    auto calc_recall_vs_bf = [&](const int64_t* result_ids, int32_t result_k, const int64_t* bf_ids, int32_t bf_k,
                                  int32_t k) {
        if (bf_ids == nullptr)
            return 0.0f;
        float total_recall = 0.0f;
        for (int q = 0; q < num_queries; ++q) {
            std::unordered_set<int64_t> bf_set;
            int bf_count = 0;
            for (int i = 0; i < k && i < bf_k; ++i) {
                if (bf_ids[q * bf_k + i] >= 0) {
                    bf_set.insert(bf_ids[q * bf_k + i]);
                    bf_count++;
                }
            }
            if (bf_count == 0)
                continue;

            int overlap = 0;
            for (int i = 0; i < k && i < result_k; ++i) {
                if (result_ids[q * result_k + i] >= 0 && bf_set.count(result_ids[q * result_k + i]) > 0) {
                    overlap++;
                }
            }
            total_recall += (float)overlap / bf_count;
        }
        return total_recall / num_queries;
    };

    // ========== BruteForce MaxSim ==========
    double bf_time = 0;
    std::vector<float> bf_recalls_vs_gt(topk_values.size(), 0.0f);
    std::map<int32_t, knowhere::DataSetPtr> bf_results;
    std::map<int32_t, const int64_t*> bf_ids_map;

    if (!SKIP_DIRECT_TEST) {
        printf("\n[BruteForce] Computing MaxSim results for each k...\n");
        fflush(stdout);

        for (int32_t k : topk_values) {
            knowhere::Json bf_conf;
            bf_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
            bf_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_bf;
            auto bf_result = knowhere::BruteForce::Search<knowhere::fp32>(doc_ds, query_ds, bf_conf, nullptr);
            bf_time += sw_bf.elapsed();
            REQUIRE(bf_result.has_value());

            bf_results[k] = bf_result.value();
            bf_ids_map[k] = bf_results[k]->GetIds();
        }
        printf("[BruteForce] Total time: %.3f s\n", bf_time);

        // Calculate GT recall at each topk
        printf("[BruteForce] Recall (vs GT): ");
        for (size_t i = 0; i < topk_values.size(); ++i) {
            int32_t k = topk_values[i];
            bf_recalls_vs_gt[i] = calc_recall_vs_gt(bf_ids_map[k], k, k);
            printf("@%d=%.1f%% ", k, bf_recalls_vs_gt[i] * 100);
        }
        printf("\n");
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

        printf("[Direct] Searching with each k...\n");
        fflush(stdout);

        for (size_t i = 0; i < topk_values.size(); ++i) {
            int32_t k = topk_values[i];
            knowhere::Json search_conf = direct_conf;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_search;
            auto direct_result = direct_index.value().Search(query_ds, search_conf, nullptr);
            direct_search_time += sw_search.elapsed();
            REQUIRE(direct_result.has_value());

            auto direct_ids = direct_result.value()->GetIds();
            direct_recalls[i] = calc_recall_vs_bf(direct_ids, k, bf_ids_map[k], k, k);
        }

        printf("[Direct] Search time: %.3f ms, Recall: ", direct_search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% ", topk_values[i], direct_recalls[i] * 100);
        }
        printf("\n");
        fflush(stdout);
    }

    // ========== MUVERA Strategy with Multiple Parameter Combinations ==========
    // Define parameter combinations: (num_projections, num_repeats)
    std::vector<std::pair<int32_t, int32_t>> muvera_params = {
        {2, 3}, {2, 5}, {2, 7}, {3, 3}, {3, 5}, {3, 7}, {4, 3}, {4, 5},
        {4, 7}, {5, 3}, {5, 5}, {5, 7}, {6, 3}, {6, 5}, {6, 7},
    };

    // Store results for each combination
    struct MuveraResult {
        int32_t num_projections;
        int32_t num_repeats;
        double build_time;
        double search_time;
        std::vector<float> recalls;  // recall at each topk
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

        printf("[MUVERA-%d-%d] Searching with each k...\n", num_proj, num_rep);
        fflush(stdout);

        std::vector<float> recalls;
        double search_time = 0;

        for (int32_t k : topk_values) {
            knowhere::Json search_conf = muvera_conf;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_muvera_search;
            auto muvera_result = muvera_index.value().Search(query_ds, search_conf, nullptr);
            search_time += sw_muvera_search.elapsed();
            REQUIRE(muvera_result.has_value());

            auto muvera_ids = muvera_result.value()->GetIds();
            float recall = bf_ids_map.count(k) > 0 ? calc_recall_vs_bf(muvera_ids, k, bf_ids_map[k], k, k) : 0.0f;
            recalls.push_back(recall);
        }

        printf("[MUVERA-%d-%d] Search time: %.3f ms, Recall: ", num_proj, num_rep, search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% ", topk_values[i], recalls[i] * 100);
        }
        printf("\n");
        fflush(stdout);

        muvera_results.push_back({num_proj, num_rep, build_time, search_time, recalls});
    }

    // ========== Summary ==========
    printf("\n============================================================================================\n");
    printf("                              Summary (MS MARCO)                                            \n");
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
        printf("BruteForce Recall (vs official GT): ");
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% ", topk_values[i], bf_recalls_vs_gt[i] * 100);
        }
        printf("\n");
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

TEST_CASE("MS MARCO ColBERT: Direct vs LEMUR", "[msmarco_emb_list_lemur]") {
    // Check if MS MARCO data files exist
    {
        std::ifstream docs_file(MSMARCO_DOCS_JSONL_PATH);
        std::ifstream queries_file(MSMARCO_QUERIES_JSONL_PATH);

        if (!docs_file.good() || !queries_file.good()) {
            printf("\n");
            printf("=============================================================\n");
            printf("MS MARCO data files not found. Please prepare the data first.\n");
            printf("Expected files:\n");
            printf("  - %s\n", MSMARCO_DOCS_JSONL_PATH.c_str());
            printf("  - %s\n", MSMARCO_QUERIES_JSONL_PATH.c_str());
            printf("\n");
            printf("Generate MS MARCO data with GT annotations:\n");
            printf("  python scripts/prepare_msmarco_with_gt.py --output-dir .\n");
            printf("=============================================================\n");
            SKIP("MS MARCO data files not found");
            return;
        }
    }

    // Load data
    printf("\n=== Loading MS MARCO Data (with GT annotations) ===\n");
    EmbListData doc_data;
    QueryDataWithGT query_data;

    REQUIRE(doc_data.LoadFromJsonl(MSMARCO_DOCS_JSONL_PATH, MAX_DOCS_TO_LOAD));
    doc_data.PrintStats();

    REQUIRE(query_data.LoadFromJsonl(MSMARCO_QUERIES_JSONL_PATH, MAX_QUERIES_TO_LOAD));
    query_data.PrintStats();

    auto doc_ds = doc_data.ToDataSet();
    auto query_ds = query_data.ToDataSet();

    const int32_t dim = doc_data.dim;
    const int32_t num_docs = doc_data.num_docs;
    const int64_t total_vectors = doc_data.total_vectors;
    const int32_t num_queries = std::min((int32_t)query_data.num_queries, MAX_QUERIES_TO_LOAD);

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

    // Base config
    knowhere::Json base_conf;
    base_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
    base_conf[knowhere::meta::DIM] = dim;
    base_conf[knowhere::meta::TOPK] = max_topk;
    base_conf[knowhere::indexparam::HNSW_M] = 16;
    base_conf[knowhere::indexparam::EFCONSTRUCTION] = 100;
    base_conf[knowhere::indexparam::EF] = 64;
    base_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 2.0f;

    auto version = GenTestEmbListVersionList();

    // Recall calculation vs BruteForce results (per-query average)
    // result_ids has result_k items per query, bf_ids has bf_k items per query
    auto calc_recall_vs_bf = [&](const int64_t* result_ids, int32_t result_k, const int64_t* bf_ids, int32_t bf_k,
                                  int32_t k) {
        if (bf_ids == nullptr)
            return 0.0f;
        float total_recall = 0.0f;
        for (int q = 0; q < num_queries; ++q) {
            std::unordered_set<int64_t> bf_set;
            int bf_count = 0;
            for (int i = 0; i < k && i < bf_k; ++i) {
                if (bf_ids[q * bf_k + i] >= 0) {
                    bf_set.insert(bf_ids[q * bf_k + i]);
                    bf_count++;
                }
            }
            if (bf_count == 0)
                continue;

            int overlap = 0;
            for (int i = 0; i < k && i < result_k; ++i) {
                if (result_ids[q * result_k + i] >= 0 && bf_set.count(result_ids[q * result_k + i]) > 0) {
                    overlap++;
                }
            }
            total_recall += (float)overlap / bf_count;
        }
        return total_recall / num_queries;
    };

    // ========== BruteForce MaxSim ==========
    // Compute BruteForce results for each k value separately
    printf("\n[BruteForce] Computing MaxSim results for each k...\n");
    fflush(stdout);

    std::map<int32_t, knowhere::DataSetPtr> bf_results;
    std::map<int32_t, const int64_t*> bf_ids_map;
    double bf_time = 0;

    for (int32_t k : topk_values) {
        knowhere::Json bf_conf;
        bf_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
        bf_conf[knowhere::meta::TOPK] = k;

        StopWatch sw_bf;
        auto bf_result = knowhere::BruteForce::Search<knowhere::fp32>(doc_ds, query_ds, bf_conf, nullptr);
        bf_time += sw_bf.elapsed();
        REQUIRE(bf_result.has_value());

        bf_results[k] = bf_result.value();
        bf_ids_map[k] = bf_results[k]->GetIds();
    }
    printf("[BruteForce] Total time: %.3f s\n", bf_time);
    fflush(stdout);

    // ========== Direct Strategy ==========
    double direct_build_time = 0, direct_search_time = 0;
    std::vector<float> direct_recalls(topk_values.size(), 0.0f);

    if (SKIP_DIRECT_TEST) {
        printf("\n[Direct] SKIPPED (SKIP_DIRECT_TEST = true)\n");
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

        printf("[Direct] Searching with each k...\n");
        fflush(stdout);

        for (size_t i = 0; i < topk_values.size(); ++i) {
            int32_t k = topk_values[i];
            knowhere::Json search_conf = direct_conf;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_search;
            auto direct_result = direct_index.value().Search(query_ds, search_conf, nullptr);
            direct_search_time += sw_search.elapsed();
            REQUIRE(direct_result.has_value());

            auto direct_ids = direct_result.value()->GetIds();
            direct_recalls[i] = calc_recall_vs_bf(direct_ids, k, bf_ids_map[k], k, k);
        }

        printf("[Direct] Search time: %.3f ms, Recall: ", direct_search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% ", topk_values[i], direct_recalls[i] * 100);
        }
        printf("\n");
        fflush(stdout);
    }

    // ========== LEMUR Strategy ==========
    // Build parameters
    const int32_t hidden_dim = 512;
    const int32_t num_layers = 2;
    const int32_t num_epochs = 30;
    const int32_t num_train_samples = 50000;

    // Search parameters to test (RETRIEVAL_ANN_RATIO)
    std::vector<float> ann_ratios = {1.3f, 1.5f, 2.0f};

    // Store results for each ratio
    struct LemurResult {
        float ann_ratio;
        double search_time;
        std::vector<float> recalls;
    };
    std::vector<LemurResult> lemur_results;
    double lemur_build_time = 0;

    printf("\n[LEMUR] Building index (h%d-l%d-e%d-s%d)...\n", hidden_dim, num_layers, num_epochs, num_train_samples);
    fflush(stdout);

    knowhere::Json lemur_conf = base_conf;
    lemur_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    lemur_conf["emb_list_strategy"] = "lemur";
    lemur_conf["lemur_hidden_dim"] = hidden_dim;
    lemur_conf["lemur_num_layers"] = num_layers;
    lemur_conf["lemur_num_epochs"] = num_epochs;
    lemur_conf["lemur_num_train_samples"] = num_train_samples;
    lemur_conf["lemur_batch_size"] = 256;
    lemur_conf["lemur_learning_rate"] = 0.001f;
    lemur_conf["lemur_seed"] = 42;
    lemur_conf["lemur_rerank"] = true;

    auto lemur_index =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
    REQUIRE(lemur_index.has_value());

    StopWatch sw_lemur_build;
    auto lemur_build_status = lemur_index.value().Build(doc_ds, lemur_conf);
    lemur_build_time = sw_lemur_build.elapsed();
    REQUIRE(lemur_build_status == knowhere::Status::success);
    printf("[LEMUR] Build time: %.3f s\n", lemur_build_time);

    // Test different ANN ratios with each k value
    printf("\n[LEMUR] Testing %zu ANN ratios x %zu k values...\n", ann_ratios.size(), topk_values.size());
    fflush(stdout);

    for (float ann_ratio : ann_ratios) {
        std::vector<float> recalls;
        double total_search_time = 0;

        for (int32_t k : topk_values) {
            knowhere::Json search_conf = lemur_conf;
            search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_lemur_search;
            auto lemur_result = lemur_index.value().Search(query_ds, search_conf, nullptr);
            double search_time = sw_lemur_search.elapsed();
            total_search_time += search_time;
            REQUIRE(lemur_result.has_value());

            auto lemur_ids = lemur_result.value()->GetIds();
            float recall = calc_recall_vs_bf(lemur_ids, k, bf_ids_map[k], k, k);
            recalls.push_back(recall);
        }

        printf("[LEMUR-ratio%.1f] Search time: %.3f ms, Recall: ", ann_ratio, total_search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% ", topk_values[i], recalls[i] * 100);
        }
        printf("\n");
        fflush(stdout);

        lemur_results.push_back({ann_ratio, total_search_time, recalls});
    }

    // ========== Summary ==========
    printf("\n=====================================================================================================\n");
    printf("                            Summary: Direct vs LEMUR (MS MARCO)                                      \n");
    printf("=====================================================================================================\n");

    // Header with topk columns
    printf("| Strategy                | Build Time | Search Time |");
    for (int32_t k : topk_values) {
        printf(" R@%-3d |", k);
    }
    printf("\n");

    printf("|-------------------------|------------|-------------|");
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf("-------|");
    }
    printf("\n");

    // BruteForce row
    printf("| BruteForce              | %10s | %9.2f s |", "-", bf_time);
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf(" %4.1f%% |", 100.0f);
    }
    printf("\n");

    // Direct row
    if (!SKIP_DIRECT_TEST) {
        printf("| Direct                  | %8.2f s | %9.2f ms |", direct_build_time, direct_search_time * 1000);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf(" %4.1f%% |", direct_recalls[i] * 100);
        }
        printf("\n");
    }

    // LEMUR rows (different ANN ratios)
    for (const auto& res : lemur_results) {
        char name[32];
        snprintf(name, sizeof(name), "LEMUR (ratio=%.1f)", res.ann_ratio);
        // First row shows build time, others show "-"
        if (&res == &lemur_results[0]) {
            printf("| %-23s | %8.2f s | %9.2f ms |", name, lemur_build_time, res.search_time * 1000);
        } else {
            printf("| %-23s | %10s | %9.2f ms |", name, "-", res.search_time * 1000);
        }
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf(" %4.1f%% |", res.recalls[i] * 100);
        }
        printf("\n");
    }

    printf("=====================================================================================================\n");
    printf("Dataset: %d docs, %ld total vectors, dim=%d, avg %.1f vectors/doc\n", num_docs, total_vectors, dim,
           (float)total_vectors / num_docs);
    printf("\nLEMUR Build Config: hidden_dim=%d, num_layers=%d, epochs=%d, train_samples=%d\n",
           hidden_dim, num_layers, num_epochs, num_train_samples);
    printf("RETRIEVAL_ANN_RATIO: Controls how many ANN candidates to retrieve (ratio * topk)\n");
    printf("\nNote: R@K = Recall at top-K, per-query averaged, compared to BruteForce MaxSim\n");
    fflush(stdout);

    // Basic sanity checks
    if (!SKIP_DIRECT_TEST) {
        for (float r : direct_recalls) {
            REQUIRE(r >= 0.0f);
        }
    }
    for (const auto& res : lemur_results) {
        for (float r : res.recalls) {
            REQUIRE(r >= 0.0f);
        }
    }
}

// ============================================================================
// Unified test for MUVERA vs LEMUR comparison with different ANN ratios
// ============================================================================
static void
RunMuveraLemurComparison(const std::string& dataset_name, const std::string& docs_path, const std::string& queries_path,
                         int32_t max_docs, int32_t max_queries) {
    // Check if data files exist
    {
        std::ifstream docs_file(docs_path);
        std::ifstream queries_file(queries_path);

        if (!docs_file.good() || !queries_file.good()) {
            printf("\n");
            printf("=============================================================\n");
            printf("%s data files not found. Please prepare the data first.\n", dataset_name.c_str());
            printf("Expected files:\n");
            printf("  - %s\n", docs_path.c_str());
            printf("  - %s\n", queries_path.c_str());
            printf("=============================================================\n");
            SKIP(dataset_name + " data files not found");
            return;
        }
    }

    // Load data
    printf("\n=== Loading %s Data ===\n", dataset_name.c_str());
    EmbListData doc_data;
    QueryDataWithGT query_data;

    REQUIRE(doc_data.LoadFromJsonl(docs_path, max_docs));
    doc_data.PrintStats();

    REQUIRE(query_data.LoadFromJsonl(queries_path, max_queries));
    query_data.PrintStats();

    auto doc_ds = doc_data.ToDataSet();
    auto query_ds = query_data.ToDataSet();

    const int32_t dim = doc_data.dim;
    const int32_t num_docs = doc_data.num_docs;
    const int64_t total_vectors = doc_data.total_vectors;
    const int32_t num_queries = std::min((int32_t)query_data.num_queries, max_queries);

    // Multiple topk values for evaluation
    const std::vector<int32_t> topk_values = {10, 20, 50};
    const std::vector<int32_t> e2e_topk_values = {1, 3, 5, 10};
    const int32_t max_topk = *std::max_element(topk_values.begin(), topk_values.end());

    // ANN ratios to test
    const std::vector<float> ann_ratios = {3.0f, 4.0f, 5.0f};

    printf("\n=== Test Configuration ===\n");
    printf("Dataset: %s\n", dataset_name.c_str());
    printf("Documents: %d, Total vectors: %ld, Dim: %d\n", num_docs, total_vectors, dim);
    printf("Queries: %d, TopK values: ", num_queries);
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf("%d%s", topk_values[i], i < topk_values.size() - 1 ? ", " : "\n");
    }
    printf("ANN Ratios to test: ");
    for (size_t i = 0; i < ann_ratios.size(); ++i) {
        printf("%.1f%s", ann_ratios[i], i < ann_ratios.size() - 1 ? ", " : "\n");
    }
    fflush(stdout);

    // Base config
    knowhere::Json base_conf;
    base_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
    base_conf[knowhere::meta::DIM] = dim;
    base_conf[knowhere::meta::TOPK] = max_topk;
    base_conf[knowhere::indexparam::HNSW_M] = 16;
    base_conf[knowhere::indexparam::EFCONSTRUCTION] = 100;
    base_conf[knowhere::indexparam::EF] = 64;

    auto version = GenTestEmbListVersionList();

    // Recall calculation vs BruteForce results
    // result_ids has result_k items per query (from searching with TOPK=result_k)
    // bf_ids has bf_k items per query (from searching with TOPK=bf_k)
    auto calc_recall_vs_bf = [&](const int64_t* result_ids, int32_t result_k, const int64_t* bf_ids, int32_t bf_k,
                                  int32_t k) {
        if (bf_ids == nullptr)
            return 0.0f;
        float total_recall = 0.0f;
        for (int q = 0; q < num_queries; ++q) {
            std::unordered_set<int64_t> bf_set;
            int bf_count = 0;
            for (int i = 0; i < k && i < bf_k; ++i) {
                if (bf_ids[q * bf_k + i] >= 0) {
                    bf_set.insert(bf_ids[q * bf_k + i]);
                    bf_count++;
                }
            }
            if (bf_count == 0)
                continue;

            int overlap = 0;
            for (int i = 0; i < k && i < result_k; ++i) {
                if (result_ids[q * result_k + i] >= 0 && bf_set.count(result_ids[q * result_k + i]) > 0) {
                    overlap++;
                }
            }
            total_recall += (float)overlap / bf_count;
        }
        return total_recall / num_queries;
    };

    // E2E Recall calculation vs Ground Truth
    // Recall@k = (number of GT documents found in top-k results) / (total GT documents)
    auto calc_recall_vs_gt = [&](const int64_t* result_ids, int32_t result_k, int32_t k) {
        float total_recall = 0.0f;
        int valid_queries = 0;
        for (int q = 0; q < num_queries; ++q) {
            const auto& gt = query_data.gt_pids[q];
            if (gt.empty())
                continue;

            std::unordered_set<int64_t> gt_set(gt.begin(), gt.end());
            int found = 0;
            for (int i = 0; i < k && i < result_k; ++i) {
                int64_t doc_id = result_ids[q * result_k + i];
                if (doc_id >= 0 && gt_set.count(doc_id) > 0) {
                    found++;
                }
            }
            total_recall += (float)found / gt.size();
            valid_queries++;
        }
        return valid_queries > 0 ? total_recall / valid_queries : 0.0f;
    };

    // ========== BruteForce MaxSim ==========
    // Compute BruteForce results for each k value separately
    printf("\n[BruteForce] Computing MaxSim results for each k...\n");
    fflush(stdout);

    std::map<int32_t, knowhere::DataSetPtr> bf_results;
    std::map<int32_t, const int64_t*> bf_ids_map;
    double bf_time = 0;

    // Compute for both topk_values and e2e_topk_values
    std::set<int32_t> all_k_values(topk_values.begin(), topk_values.end());
    all_k_values.insert(e2e_topk_values.begin(), e2e_topk_values.end());

    for (int32_t k : all_k_values) {
        knowhere::Json bf_conf;
        bf_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
        bf_conf[knowhere::meta::TOPK] = k;

        StopWatch sw_bf;
        auto bf_result = knowhere::BruteForce::Search<knowhere::fp32>(doc_ds, query_ds, bf_conf, nullptr);
        bf_time += sw_bf.elapsed();
        REQUIRE(bf_result.has_value());

        bf_results[k] = bf_result.value();
        bf_ids_map[k] = bf_results[k]->GetIds();
    }
    printf("[BruteForce] Total time: %.3f s\n", bf_time);
    fflush(stdout);

    // Result structure
    struct SearchResult {
        std::string name;
        float ann_ratio;
        double build_time;
        double search_time;
        std::vector<float> recalls;           // vs BruteForce
        std::vector<double> math_latencies;   // avg latency per query for each Math Recall topk (ms)
        std::vector<float> e2e_recalls;       // vs Ground Truth
        std::vector<double> e2e_latencies;    // avg latency per query for each E2E topk (ms)
    };
    std::vector<SearchResult> all_results;

    // BruteForce E2E recalls
    std::vector<float> bf_e2e_recalls;
    for (int32_t k : e2e_topk_values) {
        bf_e2e_recalls.push_back(calc_recall_vs_gt(bf_ids_map[k], k, k));
    }

    // ========== Direct Strategy ==========
    if (!SKIP_DIRECT_TEST) {
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
        double direct_build_time = sw_direct_build.elapsed();
        REQUIRE(direct_build_status == knowhere::Status::success);
        printf("[Direct] Build time: %.3f s\n", direct_build_time);

        // Test Direct with different ANN ratios and k values
        printf("\n[Direct] Testing %zu ANN ratios x %zu k values...\n", ann_ratios.size(), topk_values.size());
        fflush(stdout);

        for (size_t idx = 0; idx < ann_ratios.size(); ++idx) {
            float ann_ratio = ann_ratios[idx];

            std::vector<float> recalls;
            std::vector<double> math_latencies;
            std::vector<float> e2e_recalls;
            double total_search_time = 0;

            // Search for Math Recall (vs BF) with individual latency (avg per query)
            for (int32_t k : topk_values) {
                knowhere::Json search_conf = direct_conf;
                search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                search_conf[knowhere::meta::TOPK] = k;

                StopWatch sw_search;
                auto result = direct_index.value().Search(query_ds, search_conf, nullptr);
                double search_time = sw_search.elapsed();
                total_search_time += search_time;
                REQUIRE(result.has_value());

                auto result_ids = result.value()->GetIds();
                float recall = calc_recall_vs_bf(result_ids, k, bf_ids_map[k], k, k);
                recalls.push_back(recall);
                math_latencies.push_back(search_time * 1000 / num_queries);  // ms per query
            }

            // Search for E2E Recall (vs GT) with individual latency (avg per query)
            std::vector<double> e2e_latencies;
            for (int32_t k : e2e_topk_values) {
                knowhere::Json search_conf = direct_conf;
                search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                search_conf[knowhere::meta::TOPK] = k;

                StopWatch sw_e2e;
                auto result = direct_index.value().Search(query_ds, search_conf, nullptr);
                double e2e_latency = sw_e2e.elapsed() * 1000 / num_queries;  // ms per query
                REQUIRE(result.has_value());

                auto result_ids = result.value()->GetIds();
                float e2e_recall = calc_recall_vs_gt(result_ids, k, k);
                e2e_recalls.push_back(e2e_recall);
                e2e_latencies.push_back(e2e_latency);
            }

            printf("[Direct-ratio%.1f] Recall: ", ann_ratio);
            for (size_t i = 0; i < topk_values.size(); ++i) {
                printf("@%d=%.1f%% (%.2fms) ", topk_values[i], recalls[i] * 100, math_latencies[i]);
            }
            printf("\n");
            printf("                  E2E Recall: ");
            for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
                printf("@%d=%.1f%% (%.2fms) ", e2e_topk_values[i], e2e_recalls[i] * 100, e2e_latencies[i]);
            }
            printf("\n");
            fflush(stdout);

            char name[32];
            snprintf(name, sizeof(name), "Direct (ratio=%.1f)", ann_ratio);
            all_results.push_back({name, ann_ratio, idx == 0 ? direct_build_time : 0, total_search_time, recalls, math_latencies, e2e_recalls, e2e_latencies});
        }
    } else {
        printf("\n[Direct] SKIPPED (SKIP_DIRECT_TEST = true)\n");
        fflush(stdout);
    }

    // ========== MUVERA Strategy ==========
    // Multiple MUVERA parameter combinations: (num_projections, num_repeats)
    std::vector<std::pair<int32_t, int32_t>> muvera_params = {{3, 7}, {4, 7}};

    printf("\n[MUVERA] Testing %zu parameter combinations...\n", muvera_params.size());
    fflush(stdout);

    for (const auto& muvera_param : muvera_params) {
        int32_t num_proj = muvera_param.first;
        int32_t num_rep = muvera_param.second;

        printf("\n[MUVERA-%d-%d] Building index...\n", num_proj, num_rep);
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
        double muvera_build_time = sw_muvera_build.elapsed();
        REQUIRE(muvera_build_status == knowhere::Status::success);
        printf("[MUVERA-%d-%d] Build time: %.3f s\n", num_proj, num_rep, muvera_build_time);

        // Test with different ANN ratios and k values
        for (size_t idx = 0; idx < ann_ratios.size(); ++idx) {
            float ann_ratio = ann_ratios[idx];

            std::vector<float> recalls;
            std::vector<double> math_latencies;
            std::vector<float> e2e_recalls;
            double total_search_time = 0;

            // Search for Math Recall (vs BF) with individual latency (avg per query)
            for (int32_t k : topk_values) {
                knowhere::Json search_conf = muvera_conf;
                search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                search_conf[knowhere::meta::TOPK] = k;

                StopWatch sw_search;
                auto result = muvera_index.value().Search(query_ds, search_conf, nullptr);
                double search_time = sw_search.elapsed();
                total_search_time += search_time;
                REQUIRE(result.has_value());

                auto result_ids = result.value()->GetIds();
                float recall = calc_recall_vs_bf(result_ids, k, bf_ids_map[k], k, k);
                recalls.push_back(recall);
                math_latencies.push_back(search_time * 1000 / num_queries);  // ms per query
            }

            // Search for E2E Recall (vs GT) with individual latency (avg per query)
            std::vector<double> e2e_latencies;
            for (int32_t k : e2e_topk_values) {
                knowhere::Json search_conf = muvera_conf;
                search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
                search_conf[knowhere::meta::TOPK] = k;

                StopWatch sw_e2e;
                auto result = muvera_index.value().Search(query_ds, search_conf, nullptr);
                double e2e_latency = sw_e2e.elapsed() * 1000 / num_queries;  // ms per query
                REQUIRE(result.has_value());

                auto result_ids = result.value()->GetIds();
                float e2e_recall = calc_recall_vs_gt(result_ids, k, k);
                e2e_recalls.push_back(e2e_recall);
                e2e_latencies.push_back(e2e_latency);
            }

            printf("[MUVERA-%d-%d-r%.1f] Recall: ", num_proj, num_rep, ann_ratio);
            for (size_t i = 0; i < topk_values.size(); ++i) {
                printf("@%d=%.1f%% (%.2fms) ", topk_values[i], recalls[i] * 100, math_latencies[i]);
            }
            printf("\n");
            printf("                    E2E Recall: ");
            for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
                printf("@%d=%.1f%% (%.2fms) ", e2e_topk_values[i], e2e_recalls[i] * 100, e2e_latencies[i]);
            }
            printf("\n");
            fflush(stdout);

            char name[48];
            snprintf(name, sizeof(name), "MUVERA-%d-%d (r=%.1f)", num_proj, num_rep, ann_ratio);
            all_results.push_back({name, ann_ratio, idx == 0 ? muvera_build_time : 0, total_search_time, recalls, math_latencies, e2e_recalls, e2e_latencies});
        }
    }

    // ========== LEMUR Strategy ==========
    const int32_t hidden_dim = 512;
    const int32_t num_layers = 2;
    const int32_t num_epochs = 30;
    const int32_t num_train_samples = 50000;

    printf("\n[LEMUR] Building index (h%d-l%d-e%d-s%d)...\n", hidden_dim, num_layers, num_epochs, num_train_samples);
    fflush(stdout);

    knowhere::Json lemur_conf = base_conf;
    lemur_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    lemur_conf["emb_list_strategy"] = "lemur";
    lemur_conf["lemur_hidden_dim"] = hidden_dim;
    lemur_conf["lemur_num_layers"] = num_layers;
    lemur_conf["lemur_num_epochs"] = num_epochs;
    lemur_conf["lemur_num_train_samples"] = num_train_samples;
    lemur_conf["lemur_batch_size"] = 256;
    lemur_conf["lemur_learning_rate"] = 0.001f;
    lemur_conf["lemur_seed"] = 42;
    lemur_conf["lemur_rerank"] = true;

    auto lemur_index =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
    REQUIRE(lemur_index.has_value());

    StopWatch sw_lemur_build;
    auto lemur_build_status = lemur_index.value().Build(doc_ds, lemur_conf);
    double lemur_build_time = sw_lemur_build.elapsed();
    REQUIRE(lemur_build_status == knowhere::Status::success);
    printf("[LEMUR] Build time: %.3f s\n", lemur_build_time);

    // Test LEMUR with different ANN ratios and k values
    printf("\n[LEMUR] Testing %zu ANN ratios x %zu k values...\n", ann_ratios.size(), topk_values.size());
    fflush(stdout);

    for (size_t idx = 0; idx < ann_ratios.size(); ++idx) {
        float ann_ratio = ann_ratios[idx];

        std::vector<float> recalls;
        std::vector<double> math_latencies;
        std::vector<float> e2e_recalls;
        double total_search_time = 0;

        // Search for Math Recall (vs BF) with individual latency (avg per query)
        for (int32_t k : topk_values) {
            knowhere::Json search_conf = lemur_conf;
            search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_search;
            auto result = lemur_index.value().Search(query_ds, search_conf, nullptr);
            double search_time = sw_search.elapsed();
            total_search_time += search_time;
            REQUIRE(result.has_value());

            auto result_ids = result.value()->GetIds();
            float recall = calc_recall_vs_bf(result_ids, k, bf_ids_map[k], k, k);
            recalls.push_back(recall);
            math_latencies.push_back(search_time * 1000 / num_queries);  // ms per query
        }

        // Search for E2E Recall (vs GT) with individual latency (avg per query)
        std::vector<double> e2e_latencies;
        for (int32_t k : e2e_topk_values) {
            knowhere::Json search_conf = lemur_conf;
            search_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = ann_ratio;
            search_conf[knowhere::meta::TOPK] = k;

            StopWatch sw_e2e;
            auto result = lemur_index.value().Search(query_ds, search_conf, nullptr);
            double e2e_latency = sw_e2e.elapsed() * 1000 / num_queries;  // ms per query
            REQUIRE(result.has_value());

            auto result_ids = result.value()->GetIds();
            float e2e_recall = calc_recall_vs_gt(result_ids, k, k);
            e2e_recalls.push_back(e2e_recall);
            e2e_latencies.push_back(e2e_latency);
        }

        printf("[LEMUR-ratio%.1f] Recall: ", ann_ratio);
        for (size_t i = 0; i < topk_values.size(); ++i) {
            printf("@%d=%.1f%% (%.2fms) ", topk_values[i], recalls[i] * 100, math_latencies[i]);
        }
        printf("\n");
        printf("                 E2E Recall: ");
        for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
            printf("@%d=%.1f%% (%.2fms) ", e2e_topk_values[i], e2e_recalls[i] * 100, e2e_latencies[i]);
        }
        printf("\n");
        fflush(stdout);

        char name[32];
        snprintf(name, sizeof(name), "LEMUR (ratio=%.1f)", ann_ratio);
        all_results.push_back({name, ann_ratio, idx == 0 ? lemur_build_time : 0, total_search_time, recalls, math_latencies, e2e_recalls, e2e_latencies});
    }

    // ========== Summary ==========
    printf("\n");
    // Calculate separator length: 25 (strategy) + 12 (build time) + (8 + 9) * topk_values.size()
    int math_separator_len = 25 + 12 + (8 + 9) * (int)topk_values.size();
    for (int i = 0; i < math_separator_len; ++i) printf("=");
    printf("\n");
    printf("                              %s: Math Recall (with Latency)                                           \n",
           dataset_name.c_str());
    for (int i = 0; i < math_separator_len; ++i) printf("=");
    printf("\n");

    // Header with R@K and Latency@K pairs
    printf("| Strategy                | Build Time |");
    for (int32_t k : topk_values) {
        printf(" R@%-3d | Lat@%-2d |", k, k);
    }
    printf("\n");

    printf("|-------------------------|------------|");
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf("-------|--------|");
    }
    printf("\n");

    // BruteForce row (no individual latency for BF)
    printf("| BruteForce              | %10s |", "-");
    for (size_t i = 0; i < topk_values.size(); ++i) {
        printf(" %4.1f%% |      - |", 100.0f);
    }
    printf("\n");

    // All results with individual latencies
    for (const auto& res : all_results) {
        if (res.build_time > 0) {
            printf("| %-23s | %8.2f s |", res.name.c_str(), res.build_time);
        } else {
            printf("| %-23s | %10s |", res.name.c_str(), "-");
        }
        for (size_t i = 0; i < topk_values.size(); ++i) {
            if (i < res.math_latencies.size()) {
                printf(" %4.1f%% | %6.2fms |", res.recalls[i] * 100, res.math_latencies[i]);
            } else {
                printf(" %4.1f%% |      - |", res.recalls[i] * 100);
            }
        }
        printf("\n");
    }

    for (int i = 0; i < math_separator_len; ++i) printf("=");
    printf("\n");
    printf("Note: R@K = Recall at top-K, per-query averaged, compared to BruteForce MaxSim, Lat@K = Avg latency per query (ms)\n\n");

    // ========== E2E Recall Summary ==========
    printf("==========================================================================================================================================\n");
    printf("                              %s: E2E Recall (with Latency)                                            \n",
           dataset_name.c_str());
    printf("==========================================================================================================================================\n");

    // Header with R@K and Latency@K pairs
    printf("| Strategy                | Build Time |");
    for (int32_t k : e2e_topk_values) {
        printf(" R@%-3d | Lat@%-2d |", k, k);
    }
    printf("\n");

    printf("|-------------------------|------------|");
    for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
        printf("-------|--------|");
    }
    printf("\n");

    // BruteForce row (no individual latency for BF)
    printf("| BruteForce              | %10s |", "-");
    for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
        printf(" %4.1f%% |      - |", bf_e2e_recalls[i] * 100);
    }
    printf("\n");

    // All results with individual latencies
    for (const auto& res : all_results) {
        if (res.build_time > 0) {
            printf("| %-23s | %8.2f s |", res.name.c_str(), res.build_time);
        } else {
            printf("| %-23s | %10s |", res.name.c_str(), "-");
        }
        for (size_t i = 0; i < e2e_topk_values.size(); ++i) {
            if (i < res.e2e_latencies.size()) {
                printf(" %4.1f%% | %6.2fms |", res.e2e_recalls[i] * 100, res.e2e_latencies[i]);
            } else {
                printf(" %4.1f%% |      - |", res.e2e_recalls[i] * 100);
            }
        }
        printf("\n");
    }

    printf("==========================================================================================================================================\n");
    printf("Note: R@K = E2E Recall at top-K (found GT / total GT), Lat@K = Avg latency per query (ms)\n\n");

    // ========== Dataset Info ==========
    printf("Dataset: %s - %d docs, %ld total vectors, dim=%d, avg %.1f vectors/doc\n", dataset_name.c_str(), num_docs,
           total_vectors, dim, (float)total_vectors / num_docs);
    if (!SKIP_DIRECT_TEST) {
        printf("Direct Config: HNSW index on all token vectors\n");
    }
    printf("MUVERA Config: tested (proj, rep) = ");
    for (size_t i = 0; i < muvera_params.size(); ++i) {
        printf("(%d,%d)%s", muvera_params[i].first, muvera_params[i].second,
               i < muvera_params.size() - 1 ? ", " : "\n");
    }
    printf("LEMUR Config: hidden_dim=%d, num_layers=%d, epochs=%d, train_samples=%d\n", hidden_dim, num_layers,
           num_epochs, num_train_samples);
    fflush(stdout);

    // Basic sanity checks
    for (const auto& res : all_results) {
        for (float r : res.recalls) {
            REQUIRE(r >= 0.0f);
        }
        for (float r : res.e2e_recalls) {
            REQUIRE(r >= 0.0f);
        }
    }
}

TEST_CASE("LoTTE: Direct vs MUVERA vs LEMUR", "[lotte_emb_list_all]") {
    RunMuveraLemurComparison("LoTTE", LOTTE_DOCS_JSONL_PATH, LOTTE_QUERIES_JSONL_PATH, MAX_DOCS_TO_LOAD,
                             MAX_QUERIES_TO_LOAD);
}

TEST_CASE("MS MARCO: Direct vs MUVERA vs LEMUR", "[msmarco_emb_list_all]") {
    RunMuveraLemurComparison("MS MARCO", MSMARCO_DOCS_JSONL_PATH, MSMARCO_QUERIES_JSONL_PATH, MAX_DOCS_TO_LOAD,
                             MAX_QUERIES_TO_LOAD);
}

TEST_CASE("SciFact: Direct vs MUVERA vs LEMUR", "[scifact_emb_list_all]") {
    RunMuveraLemurComparison("SciFact", SCIFACT_DOCS_JSONL_PATH, SCIFACT_QUERIES_JSONL_PATH, MAX_DOCS_TO_LOAD,
                             MAX_QUERIES_TO_LOAD);
}
