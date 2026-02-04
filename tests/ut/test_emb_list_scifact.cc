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
constexpr int32_t MAX_DOCS_TO_LOAD = 6000;    // SciFact has ~5183 docs
constexpr int32_t MAX_QUERIES_TO_LOAD = 100;  // Hard-coded limit on queries
constexpr bool SKIP_DIRECT_TEST = false;      // Set to true to skip Direct strategy (it's slow)

// SciFact data file paths (with official ground truth annotations)
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
        int64_t p90_count = counts[(int64_t)(num_docs * 0.9)];

        printf("Vectors per doc: min=%ld, max=%ld, avg=%.1f, median=%ld, P90=%ld\n", min_count, max_count, avg_count,
               median_count, p90_count);
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

TEST_CASE("SciFact ColBERT: Direct vs MUVERA", "[scifact_emb_list]") {
    // Check if SciFact data files exist
    {
        std::ifstream docs_file(SCIFACT_DOCS_JSONL_PATH);
        std::ifstream queries_file(SCIFACT_QUERIES_JSONL_PATH);

        if (!docs_file.good() || !queries_file.good()) {
            printf("\n");
            printf("=============================================================\n");
            printf("SciFact data files not found. Please prepare the data first.\n");
            printf("Expected files:\n");
            printf("  - %s\n", SCIFACT_DOCS_JSONL_PATH.c_str());
            printf("  - %s\n", SCIFACT_QUERIES_JSONL_PATH.c_str());
            printf("\n");
            printf("Generate SciFact data with GT annotations:\n");
            printf("  python scripts/prepare_scifact_with_gt.py --output-dir .\n");
            printf("=============================================================\n");
            SKIP("SciFact data files not found");
            return;
        }
    }

    // Load data
    printf("\n=== Loading SciFact Data (with GT annotations) ===\n");
    EmbListData doc_data;
    QueryDataWithGT query_data;

    REQUIRE(doc_data.LoadFromJsonl(SCIFACT_DOCS_JSONL_PATH, MAX_DOCS_TO_LOAD));
    doc_data.PrintStats();

    REQUIRE(query_data.LoadFromJsonl(SCIFACT_QUERIES_JSONL_PATH, MAX_QUERIES_TO_LOAD));
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
    base_conf[knowhere::indexparam::EFCONSTRUCTION] = 100;
    base_conf[knowhere::indexparam::EF] = 64;
    base_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 2.0f;

    auto version = GenTestEmbListVersionList();

    // ========== Official Ground Truth ==========
    printf("\n[Ground Truth] Using official SciFact annotations (gt_pids)\n");
    fflush(stdout);

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

        // Calculate GT recall at each topk
        printf("[BruteForce] Recall (vs GT): ");
        for (size_t i = 0; i < topk_values.size(); ++i) {
            bf_recalls_vs_gt[i] = calc_recall_vs_gt(bf_ids, topk_values[i]);
            printf("@%d=%.1f%% ", topk_values[i], bf_recalls_vs_gt[i] * 100);
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

        printf("[MUVERA-%d-%d] Searching...\n", num_proj, num_rep);
        fflush(stdout);
        StopWatch sw_muvera_search;
        auto muvera_result = muvera_index.value().Search(query_ds, muvera_conf, nullptr);
        double search_time = sw_muvera_search.elapsed();
        REQUIRE(muvera_result.has_value());
        printf("[MUVERA-%d-%d] Search time: %.3f ms\n", num_proj, num_rep, search_time * 1000);

        auto muvera_ids = muvera_result.value()->GetIds();
        auto recalls = calc_recalls_vs_bf(muvera_ids, bf_ids);
        if (bf_ids != nullptr) {
            printf("[MUVERA-%d-%d] Recall (vs BF): ", num_proj, num_rep);
            for (size_t i = 0; i < topk_values.size(); ++i) {
                printf("@%d=%.1f%% ", topk_values[i], recalls[i] * 100);
            }
            printf("\n");
        }
        fflush(stdout);

        muvera_results.push_back({num_proj, num_rep, build_time, search_time, recalls});
    }

    // ========== Summary ==========
    printf("\n============================================================================================\n");
    printf("                              Summary (SciFact)                                             \n");
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
