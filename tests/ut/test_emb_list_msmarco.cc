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
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace {

// ============================================================================
// Configuration: Control test behavior
// ============================================================================
constexpr int32_t MAX_DOCS_TO_LOAD = 10000;     // Hard-coded limit on documents
constexpr int32_t MAX_QUERIES_TO_LOAD = 100;    // Hard-coded limit on queries
constexpr bool SKIP_DIRECT_TEST = false;        // Set to true to skip Direct strategy (it's slow)

// Data file paths (relative to build directory or absolute)
const std::string DOC_VECTORS_PATH = "msmarco_doc_vectors.bin";
const std::string DOC_OFFSETS_PATH = "msmarco_doc_offsets.bin";
const std::string QUERY_VECTORS_PATH = "msmarco_query_vectors.bin";
const std::string QUERY_OFFSETS_PATH = "msmarco_query_offsets.bin";

// ============================================================================
// Binary file format:
// - vectors.bin: [int32_t dim][int64_t total_vectors][float * dim * total_vectors]
// - offsets.bin: [int64_t num_docs][size_t * (num_docs + 1)]
// ============================================================================

struct EmbListData {
    std::vector<float> vectors;
    std::vector<size_t> offsets;
    int32_t dim = 0;
    int64_t num_docs = 0;
    int64_t total_vectors = 0;

    bool
    Load(const std::string& vectors_path, const std::string& offsets_path, int32_t max_docs) {
        // Load offsets first to determine structure
        std::ifstream ofs_file(offsets_path, std::ios::binary);
        if (!ofs_file) {
            printf("Cannot open offsets file: %s\n", offsets_path.c_str());
            return false;
        }

        int64_t file_num_docs;
        ofs_file.read(reinterpret_cast<char*>(&file_num_docs), sizeof(int64_t));
        num_docs = std::min(file_num_docs, (int64_t)max_docs);
        printf("File has %ld docs, loading %ld docs\n", file_num_docs, num_docs);

        offsets.resize(num_docs + 1);
        ofs_file.read(reinterpret_cast<char*>(offsets.data()), (num_docs + 1) * sizeof(size_t));
        ofs_file.close();

        total_vectors = offsets[num_docs];
        printf("Total vectors to load: %ld\n", total_vectors);

        // Load vectors
        std::ifstream vec_file(vectors_path, std::ios::binary);
        if (!vec_file) {
            printf("Cannot open vectors file: %s\n", vectors_path.c_str());
            return false;
        }

        vec_file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        int64_t file_total_vectors;
        vec_file.read(reinterpret_cast<char*>(&file_total_vectors), sizeof(int64_t));
        printf("File dim: %d, file total vectors: %ld\n", dim, file_total_vectors);

        if (total_vectors > file_total_vectors) {
            printf("Error: requested vectors (%ld) > file vectors (%ld)\n", total_vectors, file_total_vectors);
            return false;
        }

        vectors.resize(total_vectors * dim);
        vec_file.read(reinterpret_cast<char*>(vectors.data()), total_vectors * dim * sizeof(float));
        vec_file.close();

        printf("Loaded %ld docs, %ld vectors, dim=%d\n", num_docs, total_vectors, dim);
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

}  // namespace

TEST_CASE("MS MARCO ColBERT: Direct vs MUVERA", "[msmarco_emb_list]") {
    // Check if data files exist
    {
        std::ifstream f(DOC_VECTORS_PATH);
        if (!f.good()) {
            printf("\n");
            printf("=============================================================\n");
            printf("MS MARCO data files not found. Please prepare the data first.\n");
            printf("Expected files:\n");
            printf("  - %s\n", DOC_VECTORS_PATH.c_str());
            printf("  - %s\n", DOC_OFFSETS_PATH.c_str());
            printf("  - %s\n", QUERY_VECTORS_PATH.c_str());
            printf("  - %s\n", QUERY_OFFSETS_PATH.c_str());
            printf("\n");
            printf("Use the Python script to download and convert MS MARCO data:\n");
            printf("  python scripts/prepare_msmarco_colbert.py\n");
            printf("=============================================================\n");
            SKIP("MS MARCO data files not found");
            return;
        }
    }

    // Load document data
    printf("\n=== Loading MS MARCO ColBERT Data ===\n");
    EmbListData doc_data, query_data;

    REQUIRE(doc_data.Load(DOC_VECTORS_PATH, DOC_OFFSETS_PATH, MAX_DOCS_TO_LOAD));
    doc_data.PrintStats();

    REQUIRE(query_data.Load(QUERY_VECTORS_PATH, QUERY_OFFSETS_PATH, MAX_QUERIES_TO_LOAD));
    query_data.PrintStats();

    auto doc_ds = doc_data.ToDataSet();
    auto query_ds = query_data.ToDataSet();

    const int32_t dim = doc_data.dim;
    const int32_t num_docs = doc_data.num_docs;
    const int64_t total_vectors = doc_data.total_vectors;
    const int32_t num_queries = query_data.num_docs;
    const int32_t topk = 50;

    printf("\n=== Test Configuration ===\n");
    printf("Documents: %d, Total vectors: %ld, Dim: %d\n", num_docs, total_vectors, dim);
    printf("Queries: %d, TopK: %d\n", num_queries, topk);
    printf("MAX_DOCS_TO_LOAD: %d\n", MAX_DOCS_TO_LOAD);
    fflush(stdout);

    // Base config
    knowhere::Json base_conf;
    base_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
    base_conf[knowhere::meta::DIM] = dim;
    base_conf[knowhere::meta::TOPK] = topk;
    base_conf[knowhere::indexparam::HNSW_M] = 16;
    base_conf[knowhere::indexparam::EFCONSTRUCTION] = 100;
    base_conf[knowhere::indexparam::EF] = 64;
    base_conf[knowhere::indexparam::RETRIEVAL_ANN_RATIO] = 3.0f;

    auto version = GenTestEmbListVersionList();

    // ========== Ground Truth ==========
    printf("\n[Ground Truth] Computing BruteForce results...\n");
    fflush(stdout);
    knowhere::Json gt_conf;
    gt_conf[knowhere::meta::METRIC_TYPE] = "MAX_SIM_IP";
    gt_conf[knowhere::meta::TOPK] = topk;

    StopWatch sw_gt;
    auto gt_result = knowhere::BruteForce::Search<knowhere::fp32>(doc_ds, query_ds, gt_conf, nullptr);
    double gt_time = sw_gt.elapsed();
    REQUIRE(gt_result.has_value());
    printf("[Ground Truth] BruteForce time: %.3f s\n", gt_time);
    fflush(stdout);

    auto gt_ids = gt_result.value()->GetIds();

    // Recall calculation lambda
    auto calc_recall = [&](const int64_t* result_ids) {
        int overlap = 0;
        for (int q = 0; q < num_queries; ++q) {
            std::unordered_set<int64_t> gt_set;
            for (int i = 0; i < topk; ++i) {
                if (gt_ids[q * topk + i] >= 0) {
                    gt_set.insert(gt_ids[q * topk + i]);
                }
            }
            for (int i = 0; i < topk; ++i) {
                if (result_ids[q * topk + i] >= 0 && gt_set.count(result_ids[q * topk + i]) > 0) {
                    overlap++;
                }
            }
        }
        return (float)overlap / (num_queries * topk);
    };

    // ========== Direct Strategy ==========
    double direct_build_time = 0, direct_search_time = 0;
    float direct_recall = 0;

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
        direct_recall = calc_recall(direct_ids);
        printf("[Direct] Recall: %.2f%%\n", direct_recall * 100);
        fflush(stdout);
    }

    // ========== MUVERA Strategy ==========
    double muvera_build_time = 0, muvera_search_time = 0;
    float muvera_recall = 0;

    printf("\n[MUVERA] Building HNSW index for %d doc vectors (FDE encoded from %ld raw vectors)...\n", num_docs,
           total_vectors);
    fflush(stdout);

    knowhere::Json muvera_conf = base_conf;
    muvera_conf[knowhere::meta::INDEX_TYPE] = knowhere::IndexEnum::INDEX_HNSW;
    muvera_conf["emb_list_strategy"] = "muvera";
    muvera_conf["muvera_num_projections"] = 4;
    muvera_conf["muvera_num_repeats"] = 3;
    muvera_conf["muvera_seed"] = 42;

    auto muvera_index =
        knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_HNSW, version);
    REQUIRE(muvera_index.has_value());

    StopWatch sw_muvera_build;
    auto muvera_build_status = muvera_index.value().Build(doc_ds, muvera_conf);
    muvera_build_time = sw_muvera_build.elapsed();
    REQUIRE(muvera_build_status == knowhere::Status::success);
    printf("[MUVERA] Build time: %.3f s\n", muvera_build_time);

    printf("[MUVERA] Searching...\n");
    fflush(stdout);
    StopWatch sw_muvera_search;
    auto muvera_result = muvera_index.value().Search(query_ds, muvera_conf, nullptr);
    muvera_search_time = sw_muvera_search.elapsed();
    REQUIRE(muvera_result.has_value());
    printf("[MUVERA] Search time: %.3f ms\n", muvera_search_time * 1000);

    auto muvera_ids = muvera_result.value()->GetIds();
    muvera_recall = calc_recall(muvera_ids);
    printf("[MUVERA] Recall: %.2f%%\n", muvera_recall * 100);
    fflush(stdout);

    // ========== Summary ==========
    printf("\n========== Summary (MS MARCO ColBERT) ==========\n");
    printf("| Strategy | Index Vectors | Build Time | Search Time | Recall |\n");
    printf("|----------|---------------|------------|-------------|--------|\n");
    if (!SKIP_DIRECT_TEST) {
        printf("| Direct   | %13ld | %8.2f s | %9.2f ms | %5.1f%% |\n", total_vectors, direct_build_time,
               direct_search_time * 1000, direct_recall * 100);
    }
    printf("| MUVERA   | %13d | %8.2f s | %9.2f ms | %5.1f%% |\n", num_docs, muvera_build_time,
           muvera_search_time * 1000, muvera_recall * 100);
    printf("=================================================\n");
    printf("Avg vectors per doc: %.1f\n", (float)total_vectors / num_docs);
    if (!SKIP_DIRECT_TEST) {
        printf("MUVERA indexes %.1fx fewer vectors\n", (float)total_vectors / num_docs);
        if (muvera_build_time > 0) {
            printf("MUVERA build is %.1fx faster\n", direct_build_time / muvera_build_time);
        }
        if (muvera_search_time > 0) {
            printf("MUVERA search is %.1fx faster\n", direct_search_time / muvera_search_time);
        }
    }
    fflush(stdout);

    // Basic sanity checks
    if (!SKIP_DIRECT_TEST) {
        REQUIRE(direct_recall >= 0.1f);
    }
    REQUIRE(muvera_recall >= 0.1f);
}
