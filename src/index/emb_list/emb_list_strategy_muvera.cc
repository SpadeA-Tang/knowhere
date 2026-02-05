// Copyright (C) 2019-2023 Zilliz. All rights reserved.
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
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

#include "knowhere/comp/index_param.h"
#include "knowhere/comp/task.h"
#include "knowhere/config.h"
#include "knowhere/index/emb_list_strategy.h"
#include "knowhere/log.h"
#include "knowhere/object.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"
#include "simd/hook.h"

namespace knowhere {

// Compute distance between two vectors based on metric type
static float
ComputeDistance(const float* x, const float* y, int32_t dim, const std::string& metric_type, bool is_cosine) {
    if (metric_type == metric::IP) {
        return faiss::fvec_inner_product(x, y, dim);
    } else if (metric_type == metric::L2) {
        return faiss::fvec_L2sqr(x, y, dim);
    } else if (is_cosine || metric_type == metric::COSINE) {
        // Cosine similarity = IP(x, y) / (||x|| * ||y||)
        float ip = faiss::fvec_inner_product(x, y, dim);
        float norm_x = std::sqrt(faiss::fvec_norm_L2sqr(x, dim));
        float norm_y = std::sqrt(faiss::fvec_norm_L2sqr(y, dim));
        if (norm_x > 0 && norm_y > 0) {
            return ip / (norm_x * norm_y);
        }
        return 0.0f;
    }
    // Default to IP
    return faiss::fvec_inner_product(x, y, dim);
}

// SimHash projects vectors onto random hyperplanes and uses the signs of
// projections to form a binary hash code, which maps to a partition index.
// Similar vectors are likely to fall into the same partition.
class SimHash {
 public:
    SimHash(int32_t dim, int32_t num_projections, int32_t seed) : dim_(dim), num_projections_(num_projections) {
        GenerateProjectionMatrix(seed);
    }

    // Compute partition index for a vector, returns value in [0, 2^num_projections)
    int32_t
    GetPartitionIndex(const float* vec) const {
        int32_t index = 0;
        for (int32_t p = 0; p < num_projections_; ++p) {
            float dot = 0.0f;
            const float* proj = projection_matrix_.data() + p * dim_;
            for (int32_t d = 0; d < dim_; ++d) {
                dot += proj[d] * vec[d];
            }
            if (dot >= 0) {
                index |= (1 << p);
            }
        }
        return index;
    }

    int32_t
    NumPartitions() const {
        return 1 << num_projections_;
    }

    int32_t
    NumProjections() const {
        return num_projections_;
    }

    const std::vector<float>&
    GetProjectionMatrix() const {
        return projection_matrix_;
    }

    void
    SetProjectionMatrix(std::vector<float>&& matrix) {
        projection_matrix_ = std::move(matrix);
    }

 private:
    void
    GenerateProjectionMatrix(int32_t seed) {
        projection_matrix_.resize(num_projections_ * dim_);

        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (int32_t i = 0; i < num_projections_ * dim_; ++i) {
            projection_matrix_[i] = dist(rng);
        }

        LOG_KNOWHERE_DEBUG_ << "SimHash: generated projection matrix [" << num_projections_ << " x " << dim_
                            << "] with seed " << seed;
    }

    int32_t dim_;
    int32_t num_projections_;
    std::vector<float> projection_matrix_;  // [num_projections * dim] row-major
};

// MUVERA uses Fixed Dimensional Encoding to convert variable-length multi-vector
// documents into fixed-length single vectors for efficient ANN retrieval,
// followed by exact MaxSim reranking.
class MuveraEmbListStrategy : public EmbListStrategy {
 public:
    std::string
    Type() const override {
        return "muvera";
    }

    expected<std::optional<DataSetPtr>>
    PrepareDataForBuild(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) override {
        // 1. Read config
        num_projections_ = config.muvera_num_projections.value();
        num_repeats_ = config.muvera_num_repeats.value();
        seed_ = config.muvera_seed.value();
        original_dim_ = dataset->GetDim();
        num_docs_ = doc_offset.num_el();

        // 2. Compute num_buckets = 2^num_projections
        num_buckets_ = 1 << num_projections_;

        // 3. Initialize SimHash instances (different seed for each repeat)
        simhash_instances_.clear();
        simhash_instances_.reserve(num_repeats_);
        for (int32_t r = 0; r < num_repeats_; ++r) {
            simhash_instances_.emplace_back(original_dim_, num_projections_, seed_ + r);
        }

        // 4. Compute encoded dimension
        encoded_dim_ = num_repeats_ * num_buckets_ * original_dim_;

        LOG_KNOWHERE_INFO_ << "MUVERA PrepareDataForBuild: num_docs=" << num_docs_ << ", original_dim=" << original_dim_
                           << ", num_projections=" << num_projections_ << ", num_buckets=" << num_buckets_
                           << ", num_repeats=" << num_repeats_ << ", encoded_dim=" << encoded_dim_;

        // 5. Allocate encoded data buffer and perform FDE encoding
        // Document uses mean aggregation (asymmetric with query's sum)
        auto encoded_data = std::make_unique<float[]>(num_docs_ * encoded_dim_);
        const float* raw_data = static_cast<const float*>(dataset->GetTensor());

        EncodeFDE(raw_data, doc_offset, encoded_data.get(), num_docs_, /*use_mean=*/true);

        LOG_KNOWHERE_INFO_ << "MUVERA FDE encoding completed";

        // 7. Store raw data and doc_offset for reranking
        emb_list_offset_ = std::make_shared<EmbListOffset>(doc_offset.offset);

        size_t total_vectors = doc_offset.offset.back();
        raw_data_.resize(total_vectors * original_dim_);
        std::memcpy(raw_data_.data(), raw_data, total_vectors * original_dim_ * sizeof(float));

        LOG_KNOWHERE_INFO_ << "MUVERA stored raw data: " << total_vectors << " vectors for reranking";

        // 8. Create encoded dataset
        auto encoded_dataset = GenDataSet(num_docs_, encoded_dim_, encoded_data.release());

        return std::optional<DataSetPtr>(encoded_dataset);
    }

    bool
    NeedsBaseIndexIDMap() const override {
        return false;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr query_dataset, const EmbListOffset& query_offset, int32_t k, const BaseConfig& config,
           const EmbListSearchContext& ctx) const override {
        // 1. Validate state
        if (!emb_list_offset_ || raw_data_.empty()) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "MUVERA not initialized");
        }

        // 2. Parse metric type
        auto metric_type = config.metric_type.value();
        auto el_metric_type_or = get_el_metric_type(metric_type);
        if (!el_metric_type_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Invalid metric type: " << metric_type;
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid metric type");
        }

        auto sub_metric_type_or = get_sub_metric_type(metric_type);
        if (!sub_metric_type_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Invalid metric type: " << metric_type;
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid metric type");
        }
        auto sub_metric_type = sub_metric_type_or.value();

        bool larger_is_closer = true;
        if (sub_metric_type == metric::L2 || sub_metric_type == metric::HAMMING || sub_metric_type == metric::JACCARD) {
            larger_is_closer = false;
        }
        bool is_cosine = (sub_metric_type == metric::COSINE);

        LOG_KNOWHERE_DEBUG_ << "MUVERA Search: metric=" << metric_type << ", sub_metric=" << sub_metric_type;

        // 3. FDE encode query documents
        // Query uses sum aggregation (asymmetric with document's mean)
        auto num_query_docs = query_offset.num_el();
        auto query_data = static_cast<const float*>(query_dataset->GetTensor());

        auto encode_start = std::chrono::high_resolution_clock::now();
        std::vector<float> encoded_queries(num_query_docs * encoded_dim_);
        EncodeFDE(query_data, query_offset, encoded_queries.data(), num_query_docs, /*use_mean=*/false);
        auto encode_end = std::chrono::high_resolution_clock::now();
        double encode_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

        LOG_KNOWHERE_INFO_ << "[MUVERA] Stage1 FDE encode: " << encode_ms << " ms, num_query_docs=" << num_query_docs;

        // 4. ANN search with encoded queries
        bool do_rerank = config.muvera_rerank.value_or(true);
        auto encoded_query_dataset = GenDataSet(num_query_docs, encoded_dim_, encoded_queries.data());

        // If reranking, retrieve more candidates; otherwise just retrieve k
        int32_t ann_k;
        if (do_rerank) {
            auto retrieval_ann_ratio = config.retrieval_ann_ratio.value();
            ann_k = std::min(std::max((int32_t)(k * retrieval_ann_ratio), 1), (int32_t)num_docs_);
        } else {
            ann_k = std::min(k, (int32_t)num_docs_);
        }

        auto ann_start = std::chrono::high_resolution_clock::now();
        auto ann_result = ctx.ann_search(encoded_query_dataset, ann_k);
        if (!ann_result.has_value()) {
            LOG_KNOWHERE_ERROR_ << "MUVERA ANN search failed: " << ann_result.what();
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "ANN search failed");
        }
        auto ann_end = std::chrono::high_resolution_clock::now();
        double ann_ms = std::chrono::duration<double, std::milli>(ann_end - ann_start).count();
        const auto* ann_ids = ann_result.value()->GetIds();

        LOG_KNOWHERE_INFO_ << "[MUVERA] Stage2 ANN search: " << ann_ms << " ms, ann_k=" << ann_k
                           << ", encoded_dim=" << encoded_dim_ << ", rerank=" << do_rerank;

        // 5. If reranking disabled, directly return ANN results
        if (!do_rerank) {
            auto ids = std::make_unique<int64_t[]>(num_query_docs * k);
            auto dists = std::make_unique<float[]>(num_query_docs * k);
            const auto* ann_dists = ann_result.value()->GetDistance();

            for (size_t q = 0; q < num_query_docs; ++q) {
                for (int32_t i = 0; i < k; ++i) {
                    ids[q * k + i] = ann_ids[q * ann_k + i];
                    dists[q * k + i] = ann_dists[q * ann_k + i];
                }
            }
            LOG_KNOWHERE_INFO_ << "[MUVERA] Reranking disabled, returning ANN results directly";
            return GenResultDataSet((int64_t)num_query_docs, (int64_t)k, std::move(ids), std::move(dists));
        }

        // 6. MaxSim reranking
        auto ids = std::make_unique<int64_t[]>(num_query_docs * k);
        auto dists = std::make_unique<float[]>(num_query_docs * k);

        auto rerank_start = std::chrono::high_resolution_clock::now();
        double total_candidate_collect_ms = 0;
        double total_distance_compute_ms = 0;
        size_t total_candidates = 0;
        size_t total_distance_computations = 0;
        size_t total_doc_vecs = 0;
        size_t total_query_vecs = 0;

        for (size_t q = 0; q < num_query_docs; ++q) {
            size_t q_vec_start = query_offset.offset[q];
            size_t q_vec_end = query_offset.offset[q + 1];
            size_t nq = q_vec_end - q_vec_start;
            total_query_vecs += nq;

            // Collect unique candidate doc IDs
            auto collect_start = std::chrono::high_resolution_clock::now();
            std::unordered_set<int64_t> candidate_docs;
            for (int32_t i = 0; i < ann_k; ++i) {
                int64_t doc_id = ann_ids[q * ann_k + i];
                if (doc_id >= 0 && doc_id < num_docs_) {
                    candidate_docs.insert(doc_id);
                }
            }
            auto collect_end = std::chrono::high_resolution_clock::now();
            total_candidate_collect_ms +=
                std::chrono::duration<double, std::milli>(collect_end - collect_start).count();
            total_candidates += candidate_docs.size();

            // Compute MaxSim score for each candidate using RerankCandidates
            auto dist_start = std::chrono::high_resolution_clock::now();

            // Pre-allocate distance matrix buffer (reused across all candidates)
            // Size: nq * max_doc_len, where max_doc_len is the maximum vectors per document
            size_t max_doc_len = 0;
            size_t query_doc_vecs = 0;
            for (int64_t doc_id : candidate_docs) {
                size_t doc_len = emb_list_offset_->offset[doc_id + 1] - emb_list_offset_->offset[doc_id];
                max_doc_len = std::max(max_doc_len, doc_len);
                query_doc_vecs += doc_len;
            }
            total_doc_vecs += query_doc_vecs;
            std::vector<float> dist_matrix(nq * max_doc_len);

            // Get thread pool for parallel distance computation (same as Direct strategy)
            auto search_pool = ThreadPool::GetGlobalSearchThreadPool();

            auto compute_score = [&](int64_t doc_id) -> std::optional<float> {
                size_t doc_vec_start = emb_list_offset_->offset[doc_id];
                size_t doc_vec_end = emb_list_offset_->offset[doc_id + 1];
                size_t doc_len = doc_vec_end - doc_vec_start;

                total_distance_computations += nq * doc_len;

                // Reuse pre-allocated distance matrix buffer
                const float* d_base = raw_data_.data() + doc_vec_start * original_dim_;

                // Parallel distance computation across query vectors (same pattern as CalcDistByIDs)
                std::vector<folly::Future<folly::Unit>> futs;
                futs.reserve(nq);

                for (size_t qi = 0; qi < nq; ++qi) {
                    futs.emplace_back(search_pool->push([&, qi = qi, doc_len = doc_len]() {
                        const float* q_vec = query_data + (q_vec_start + qi) * original_dim_;

                        if (sub_metric_type == metric::IP) {
                            // Use SIMD-optimized batch inner product
                            faiss::fvec_inner_products_ny(dist_matrix.data() + qi * doc_len, q_vec, d_base,
                                                          original_dim_, doc_len);
                        } else if (sub_metric_type == metric::L2) {
                            // Use SIMD-optimized batch L2 distance
                            faiss::fvec_L2sqr_ny(dist_matrix.data() + qi * doc_len, q_vec, d_base, original_dim_,
                                                 doc_len);
                        } else if (is_cosine) {
                            // For COSINE: normalize query vector and compute IP with normalized doc vectors
                            float q_norm = faiss::fvec_norm_L2sqr(q_vec, original_dim_);
                            q_norm = (q_norm > 0) ? 1.0f / std::sqrt(q_norm) : 0.0f;

                            for (size_t di = 0; di < doc_len; ++di) {
                                const float* d_vec = d_base + di * original_dim_;
                                float d_norm = faiss::fvec_norm_L2sqr(d_vec, original_dim_);
                                d_norm = (d_norm > 0) ? 1.0f / std::sqrt(d_norm) : 0.0f;
                                float ip = faiss::fvec_inner_product(q_vec, d_vec, original_dim_);
                                dist_matrix[qi * doc_len + di] = ip * q_norm * d_norm;
                            }
                        } else {
                            // Fallback: use ComputeDistance for other metrics
                            for (size_t di = 0; di < doc_len; ++di) {
                                const float* d_vec = d_base + di * original_dim_;
                                dist_matrix[qi * doc_len + di] =
                                    ComputeDistance(q_vec, d_vec, original_dim_, sub_metric_type, is_cosine);
                            }
                        }
                    }));
                }

                // Wait for all threads to complete (same as CalcDistByIDs)
                WaitAllSuccess(futs);

                return get_sum_max_sim(dist_matrix.data(), nq, doc_len, larger_is_closer);
            };

            RerankCandidates(candidate_docs, k, larger_is_closer, compute_score, ids.get() + q * k,
                             dists.get() + q * k);
            auto dist_end = std::chrono::high_resolution_clock::now();
            total_distance_compute_ms += std::chrono::duration<double, std::milli>(dist_end - dist_start).count();
        }

        auto rerank_end = std::chrono::high_resolution_clock::now();
        double rerank_ms = std::chrono::duration<double, std::milli>(rerank_end - rerank_start).count();

        double avg_doc_len = total_candidates > 0 ? (double)total_doc_vecs / total_candidates : 0;
        double avg_query_len = num_query_docs > 0 ? (double)total_query_vecs / num_query_docs : 0;
        LOG_KNOWHERE_INFO_ << "[MUVERA] Stage3 Rerank: " << rerank_ms << " ms"
                           << ", candidate_collect=" << total_candidate_collect_ms << " ms"
                           << ", distance_compute=" << total_distance_compute_ms << " ms"
                           << ", total_candidates=" << total_candidates << ", avg_doc_len=" << avg_doc_len
                           << ", avg_query_len=" << avg_query_len
                           << ", total_dist_comps=" << total_distance_computations;

        return GenResultDataSet((int64_t)num_query_docs, (int64_t)k, std::move(ids), std::move(dists));
    }

    Status
    Serialize(BinarySet& binset) const override {
        // 1. Serialize config parameters
        // Format: [num_projections, num_repeats, seed, original_dim, num_docs] (5 x int32/int64)
        size_t config_size = 4 * sizeof(int32_t) + sizeof(int64_t);
        auto config_data = std::shared_ptr<uint8_t[]>(new uint8_t[config_size]);
        uint8_t* ptr = config_data.get();

        std::memcpy(ptr, &num_projections_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_repeats_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &seed_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &original_dim_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_docs_, sizeof(int64_t));

        binset.Append(meta::MUVERA_CONFIG, config_data, config_size);

        // 2. Serialize document offsets (same format as Direct strategy)
        if (emb_list_offset_) {
            size_t num_offsets = emb_list_offset_->offset.size();
            size_t offset_size = sizeof(size_t) + num_offsets * sizeof(size_t);
            auto offset_data = std::shared_ptr<uint8_t[]>(new uint8_t[offset_size]);

            std::memcpy(offset_data.get(), &num_offsets, sizeof(size_t));
            std::memcpy(offset_data.get() + sizeof(size_t), emb_list_offset_->offset.data(),
                        num_offsets * sizeof(size_t));

            binset.Append(meta::EMB_LIST_META, offset_data, offset_size);
        }

        // 3. Serialize raw data for reranking
        if (!raw_data_.empty()) {
            size_t raw_size = raw_data_.size() * sizeof(float);
            auto raw_data_bin = std::shared_ptr<uint8_t[]>(new uint8_t[raw_size]);
            std::memcpy(raw_data_bin.get(), raw_data_.data(), raw_size);

            binset.Append(meta::MUVERA_RAW_DATA, raw_data_bin, raw_size);
        }

        LOG_KNOWHERE_INFO_ << "MUVERA Serialize: config + offsets + raw_data";
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, const BaseConfig& config) override {
        // 1. Deserialize config parameters
        auto config_bin = binset.GetByName(meta::MUVERA_CONFIG);
        if (!config_bin) {
            LOG_KNOWHERE_WARNING_ << "No MUVERA config found in binary set";
            return Status::emb_list_inner_error;
        }

        const uint8_t* ptr = config_bin->data.get();
        std::memcpy(&num_projections_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&num_repeats_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&seed_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&original_dim_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&num_docs_, ptr, sizeof(int64_t));

        // 2. Recompute derived values
        num_buckets_ = 1 << num_projections_;
        encoded_dim_ = num_repeats_ * num_buckets_ * original_dim_;

        // 3. Rebuild SimHash instances with the same seeds
        simhash_instances_.clear();
        simhash_instances_.reserve(num_repeats_);
        for (int32_t r = 0; r < num_repeats_; ++r) {
            simhash_instances_.emplace_back(original_dim_, num_projections_, seed_ + r);
        }

        LOG_KNOWHERE_INFO_ << "MUVERA Deserialize config: num_projections=" << num_projections_
                           << ", num_repeats=" << num_repeats_ << ", seed=" << seed_
                           << ", original_dim=" << original_dim_ << ", num_docs=" << num_docs_
                           << ", encoded_dim=" << encoded_dim_;

        // 4. Deserialize document offsets
        auto offset_bin = binset.GetByName(meta::EMB_LIST_META);
        if (!offset_bin) {
            LOG_KNOWHERE_WARNING_ << "No emb_list offset found in binary set";
            return Status::emb_list_inner_error;
        }

        size_t num_offsets = 0;
        std::memcpy(&num_offsets, offset_bin->data.get(), sizeof(size_t));
        std::vector<size_t> offset(num_offsets);
        std::memcpy(offset.data(), offset_bin->data.get() + sizeof(size_t), num_offsets * sizeof(size_t));
        emb_list_offset_ = std::make_shared<EmbListOffset>(std::move(offset));

        // 5. Deserialize raw data for reranking
        auto raw_bin = binset.GetByName(meta::MUVERA_RAW_DATA);
        if (!raw_bin) {
            LOG_KNOWHERE_WARNING_ << "No MUVERA raw data found in binary set";
            return Status::emb_list_inner_error;
        }

        size_t num_floats = raw_bin->size / sizeof(float);
        raw_data_.resize(num_floats);
        std::memcpy(raw_data_.data(), raw_bin->data.get(), raw_bin->size);

        LOG_KNOWHERE_INFO_ << "MUVERA Deserialize: loaded " << num_floats << " floats for reranking";

        return Status::success;
    }

    int32_t
    GetIndexedDim() const override {
        return encoded_dim_;
    }

    int64_t
    GetDocCount() const override {
        return num_docs_;
    }

    std::shared_ptr<EmbListOffset>
    GetEmbListOffset() const override {
        return emb_list_offset_;
    }

 private:
    int32_t num_projections_ = 0;
    int32_t num_buckets_ = 0;  // = 2^num_projections
    int32_t num_repeats_ = 0;
    int32_t seed_ = 0;
    int32_t original_dim_ = 0;
    int32_t encoded_dim_ = 0;  // = num_repeats * num_buckets * original_dim
    int64_t num_docs_ = 0;

    std::vector<SimHash> simhash_instances_;  // one per repetition

    // Storage for reranking
    std::shared_ptr<EmbListOffset> emb_list_offset_;  // document offsets
    std::vector<float> raw_data_;                     // original vectors for MaxSim rerank

    // FDE encoding: converts multi-vector documents to fixed-length vectors
    // @param use_mean: if true, use mean aggregation; if false, use sum aggregation
    void
    EncodeFDE(const float* raw_data, const EmbListOffset& offset, float* encoded_data, int64_t num_items,
              bool use_mean) const {
        std::memset(encoded_data, 0, num_items * encoded_dim_ * sizeof(float));

        // Bucket counts for mean aggregation: [num_items, num_repeats, num_buckets]
        std::vector<int32_t> bucket_counts;
        if (use_mean) {
            bucket_counts.resize(num_items * num_repeats_ * num_buckets_, 0);
        }

        for (int64_t item_id = 0; item_id < num_items; ++item_id) {
            size_t vec_start = offset.offset[item_id];
            size_t vec_end = offset.offset[item_id + 1];
            float* item_encoded = encoded_data + item_id * encoded_dim_;
            int32_t* item_counts = use_mean ? bucket_counts.data() + item_id * num_repeats_ * num_buckets_ : nullptr;

            for (int32_t r = 0; r < num_repeats_; ++r) {
                float* repeat_encoded = item_encoded + r * num_buckets_ * original_dim_;
                int32_t* repeat_counts = use_mean ? item_counts + r * num_buckets_ : nullptr;

                for (size_t vec_idx = vec_start; vec_idx < vec_end; ++vec_idx) {
                    const float* vec = raw_data + vec_idx * original_dim_;
                    int32_t bucket_idx = simhash_instances_[r].GetPartitionIndex(vec);
                    float* bucket = repeat_encoded + bucket_idx * original_dim_;

                    for (int32_t d = 0; d < original_dim_; ++d) {
                        bucket[d] += vec[d];
                    }
                    if (use_mean) {
                        repeat_counts[bucket_idx]++;
                    }
                }

                // Mean aggregation: divide by count
                if (use_mean) {
                    for (int32_t b = 0; b < num_buckets_; ++b) {
                        if (repeat_counts[b] > 1) {
                            float* bucket = repeat_encoded + b * original_dim_;
                            float inv_count = 1.0f / repeat_counts[b];
                            for (int32_t d = 0; d < original_dim_; ++d) {
                                bucket[d] *= inv_count;
                            }
                        }
                    }
                }
            }
        }
    }
};

EmbListStrategyPtr
CreateMuveraEmbListStrategy() {
    return std::make_unique<MuveraEmbListStrategy>();
}

}  // namespace knowhere
