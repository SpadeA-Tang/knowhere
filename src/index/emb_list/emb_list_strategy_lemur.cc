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

#include <cblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
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
#include "simple_mlp.h"
#include "simd/hook.h"

namespace knowhere {

// Find max value in an array
static inline float
FindMax(const float* data, size_t len) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < len; ++i) {
        max_val = std::max(max_val, data[i]);
    }
    return max_val;
}

/**
 * @brief LEMUR (Learned Multi-Vector Retrieval) strategy.
 *
 * LEMUR learns a neural network to compress multi-vector documents into
 * fixed-dimensional representations. Unlike MUVERA's random projections,
 * LEMUR is data-aware and can adapt to the corpus distribution.
 *
 * Training:
 *   1. Sample vectors from corpus as training inputs
 *   2. Compute MaxSim between sampled vectors and all documents as labels
 *   3. Train MLP: input(dim) -> hidden(hidden_dim) -> output(num_docs)
 *   4. W matrix = output_layer weights, shape [num_docs, hidden_dim]
 *
 * Search:
 *   1. Extract query features: feature_extractor(query_vectors)
 *   2. Aggregate query features (sum/mean)
 *   3. Approximate scoring: query_feat @ W.T
 *   4. ANN search on W to find candidates
 *   5. MaxSim reranking on candidates
 */
class LemurEmbListStrategy : public EmbListStrategy {
 public:
    std::string
    Type() const override {
        return "lemur";
    }

    expected<std::optional<DataSetPtr>>
    PrepareDataForBuild(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) override {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 1. Read config
        hidden_dim_ = config.lemur_hidden_dim.value();
        num_train_samples_ = config.lemur_num_train_samples.value();
        num_epochs_ = config.lemur_num_epochs.value();
        batch_size_ = config.lemur_batch_size.value();
        learning_rate_ = config.lemur_learning_rate.value();
        seed_ = config.lemur_seed.value();
        num_layers_ = config.lemur_num_layers.value();

        original_dim_ = dataset->GetDim();
        num_docs_ = doc_offset.num_el();
        size_t total_vectors = doc_offset.offset.back();

        LOG_KNOWHERE_INFO_ << "LEMUR PrepareDataForBuild: num_docs=" << num_docs_ << ", total_vectors=" << total_vectors
                           << ", original_dim=" << original_dim_ << ", hidden_dim=" << hidden_dim_
                           << ", num_train_samples=" << num_train_samples_ << ", epochs=" << num_epochs_;

        // 2. Store raw data and doc_offset for reranking
        emb_list_offset_ = std::make_shared<EmbListOffset>(doc_offset.offset);
        const float* raw_data = static_cast<const float*>(dataset->GetTensor());
        raw_data_.resize(total_vectors * original_dim_);
        std::memcpy(raw_data_.data(), raw_data, total_vectors * original_dim_ * sizeof(float));

        // 3. Sample training vectors
        std::mt19937 rng(seed_);
        int32_t actual_samples = std::min(num_train_samples_, (int32_t)total_vectors);
        std::vector<int32_t> sample_indices(total_vectors);
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
        sample_indices.resize(actual_samples);

        std::vector<float> X_train(actual_samples * original_dim_);
        for (int32_t i = 0; i < actual_samples; ++i) {
            std::memcpy(X_train.data() + i * original_dim_, raw_data + sample_indices[i] * original_dim_,
                        original_dim_ * sizeof(float));
        }

        LOG_KNOWHERE_INFO_ << "LEMUR: Sampled " << actual_samples << " vectors for training";

        // 4. Compute training labels (MaxSim for each sample vector against each document)
        auto label_start = std::chrono::high_resolution_clock::now();
        std::vector<float> y_train(actual_samples * num_docs_);
        ComputeMaxSimLabels(X_train.data(), actual_samples, raw_data, doc_offset, y_train.data());
        auto label_end = std::chrono::high_resolution_clock::now();
        double label_ms = std::chrono::duration<double, std::milli>(label_end - label_start).count();

        LOG_KNOWHERE_INFO_ << "LEMUR: Computed MaxSim labels in " << label_ms << " ms";

        // 5. Save raw labels for OLS (original LEMUR uses raw MaxSim, not normalized)
        std::vector<float> y_train_raw = y_train;

        // 6. Normalize labels (z-score normalization for stable MLP training)
        float label_mean = 0.0f, label_std = 0.0f;
        size_t label_count = actual_samples * num_docs_;
        for (size_t i = 0; i < label_count; ++i) {
            label_mean += y_train[i];
        }
        label_mean /= label_count;

        for (size_t i = 0; i < label_count; ++i) {
            float diff = y_train[i] - label_mean;
            label_std += diff * diff;
        }
        label_std = std::sqrt(label_std / label_count);
        if (label_std < 1e-6f) {
            label_std = 1.0f;
        }

        for (size_t i = 0; i < label_count; ++i) {
            y_train[i] = (y_train[i] - label_mean) / label_std;
        }
        label_mean_ = label_mean;
        label_std_ = label_std;

        LOG_KNOWHERE_INFO_ << "LEMUR: Label normalization - mean=" << label_mean_ << ", std=" << label_std_;

        // 6. Train MLP
        // SimpleMLP(input_dim, output_dim, hidden_dim, final_hidden_dim, num_layers, seed)
        auto train_start = std::chrono::high_resolution_clock::now();
        mlp_ = std::make_unique<SimpleMLP>(original_dim_, num_docs_, hidden_dim_, hidden_dim_, num_layers_, seed_);
        float final_loss = mlp_->Train(X_train.data(), y_train.data(), actual_samples, num_epochs_, batch_size_,
                                       learning_rate_, /*verbose=*/true);
        auto train_end = std::chrono::high_resolution_clock::now();
        double train_ms = std::chrono::duration<double, std::milli>(train_end - train_start).count();

        LOG_KNOWHERE_INFO_ << "LEMUR: MLP training completed in " << train_ms << " ms, final_loss=" << final_loss;

        // 7. Compute W using pseudoinverse (fit_corpus step from original LEMUR)
        // W = pinv(Z) @ Y, where Z = features of sampled vectors, Y = MaxSim labels
        // This is a least-squares solution: find W such that Z @ W^T ≈ Y
        auto ols_start = std::chrono::high_resolution_clock::now();
        final_hidden_dim_ = mlp_->FinalHiddenDim();

        // Extract features for all training samples: Z [num_samples, hidden_dim]
        std::vector<float> Z(actual_samples * final_hidden_dim_);
        mlp_->ExtractFeatures(X_train.data(), actual_samples, Z.data());

        // Compute W = pinv(Z) @ Y using normal equations: W = (Z^T Z)^{-1} Z^T Y
        // For numerical stability, we solve the normal equations directly
        // Z^T Z: [hidden_dim, hidden_dim]
        // Z^T Y: [hidden_dim, num_docs]
        std::vector<float> ZtZ(final_hidden_dim_ * final_hidden_dim_, 0.0f);
        std::vector<float> ZtY(final_hidden_dim_ * num_docs_, 0.0f);

        // Compute Z^T @ Z using BLAS
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, final_hidden_dim_, final_hidden_dim_, actual_samples, 1.0f,
                    Z.data(), final_hidden_dim_, Z.data(), final_hidden_dim_, 0.0f, ZtZ.data(), final_hidden_dim_);

        // Add regularization for numerical stability: ZtZ += lambda * I
        const float lambda = 1e-4f;
        for (int32_t i = 0; i < final_hidden_dim_; ++i) {
            ZtZ[i * final_hidden_dim_ + i] += lambda;
        }

        // Compute Z^T @ Y using BLAS (Y is raw MaxSim labels, matching original LEMUR)
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, final_hidden_dim_, num_docs_, actual_samples, 1.0f,
                    Z.data(), final_hidden_dim_, y_train_raw.data(), num_docs_, 0.0f, ZtY.data(), num_docs_);

        // Solve (Z^T Z) @ W^T = Z^T Y using Cholesky decomposition
        // Since ZtZ is symmetric positive definite (with regularization), use Cholesky
        // L @ L^T = ZtZ, then solve L @ L^T @ X = ZtY

        // Cholesky decomposition: ZtZ = L @ L^T (in-place, lower triangular)
        bool cholesky_ok = true;
        for (int32_t i = 0; i < final_hidden_dim_; ++i) {
            for (int32_t j = 0; j <= i; ++j) {
                float sum = ZtZ[i * final_hidden_dim_ + j];
                for (int32_t k = 0; k < j; ++k) {
                    sum -= ZtZ[i * final_hidden_dim_ + k] * ZtZ[j * final_hidden_dim_ + k];
                }
                if (i == j) {
                    if (sum <= 0.0f) {
                        cholesky_ok = false;
                        break;
                    }
                    ZtZ[i * final_hidden_dim_ + i] = std::sqrt(sum);
                } else {
                    ZtZ[i * final_hidden_dim_ + j] = sum / ZtZ[j * final_hidden_dim_ + j];
                }
            }
            if (!cholesky_ok)
                break;
        }

        if (!cholesky_ok) {
            LOG_KNOWHERE_WARNING_ << "LEMUR: Cholesky decomposition failed, falling back to MLP output weights";
            const auto& W_out = mlp_->GetOutputWeights();
            W_.resize(num_docs_ * final_hidden_dim_);
            std::memcpy(W_.data(), W_out.data(), W_.size() * sizeof(float));
        } else {
            // Solve L @ L^T @ X = ZtY using BLAS triangular solve
            // Step 1: Solve L @ Y = ZtY (forward substitution)
            // Step 2: Solve L^T @ X = Y (backward substitution)

            // Copy ZtY to Wt (will be overwritten with solution)
            std::vector<float> Wt(final_hidden_dim_ * num_docs_);
            std::memcpy(Wt.data(), ZtY.data(), Wt.size() * sizeof(float));

            // cblas_strsm: solve op(A) @ X = alpha * B, result stored in B
            // L is lower triangular, stored in ZtZ (row-major)
            // Wt is [hidden_dim, num_docs] (row-major)

            // Step 1: L @ Y = Wt => Y = L^{-1} @ Wt
            cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, final_hidden_dim_, num_docs_,
                        1.0f, ZtZ.data(), final_hidden_dim_, Wt.data(), num_docs_);

            // Step 2: L^T @ X = Y => X = L^{-T} @ Y
            cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, final_hidden_dim_, num_docs_,
                        1.0f, ZtZ.data(), final_hidden_dim_, Wt.data(), num_docs_);

            // Transpose W^T [hidden_dim, num_docs] to get W [num_docs, hidden_dim]
            W_.resize(num_docs_ * final_hidden_dim_);
            for (int64_t d = 0; d < num_docs_; ++d) {
                for (int32_t h = 0; h < final_hidden_dim_; ++h) {
                    W_[d * final_hidden_dim_ + h] = Wt[h * num_docs_ + d];
                }
            }
        }

        auto ols_end = std::chrono::high_resolution_clock::now();
        double ols_ms = std::chrono::duration<double, std::milli>(ols_end - ols_start).count();

        LOG_KNOWHERE_INFO_ << "LEMUR: Computed W via pseudoinverse (OLS) [" << num_docs_ << " x " << final_hidden_dim_
                           << "] in " << ols_ms << " ms";

        // 8. Create dataset from W for ANN indexing
        auto W_data = std::make_unique<float[]>(num_docs_ * final_hidden_dim_);
        std::memcpy(W_data.get(), W_.data(), num_docs_ * final_hidden_dim_ * sizeof(float));
        auto W_dataset = GenDataSet(num_docs_, final_hidden_dim_, W_data.release());

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        LOG_KNOWHERE_INFO_ << "LEMUR PrepareDataForBuild completed in " << total_ms << " ms";

        return std::optional<DataSetPtr>(W_dataset);
    }

    bool
    NeedsBaseIndexIDMap() const override {
        return false;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr query_dataset, const EmbListOffset& query_offset, int32_t k, const BaseConfig& config,
           const EmbListSearchContext& ctx) const override {
        // 1. Validate state
        if (!mlp_ || !emb_list_offset_ || raw_data_.empty() || W_.empty()) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "LEMUR not initialized");
        }

        // 2. Parse metric type
        auto metric_type = config.metric_type.value();
        auto el_metric_type_or = get_el_metric_type(metric_type);
        if (!el_metric_type_or.has_value()) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid metric type");
        }

        auto sub_metric_type_or = get_sub_metric_type(metric_type);
        if (!sub_metric_type_or.has_value()) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid metric type");
        }
        auto sub_metric_type = sub_metric_type_or.value();

        bool larger_is_closer = true;
        if (sub_metric_type == metric::L2 || sub_metric_type == metric::HAMMING || sub_metric_type == metric::JACCARD) {
            larger_is_closer = false;
        }
        bool is_cosine = (sub_metric_type == metric::COSINE);

        auto num_query_docs = query_offset.num_el();
        const float* query_data = static_cast<const float*>(query_dataset->GetTensor());

        LOG_KNOWHERE_DEBUG_ << "LEMUR Search: num_query_docs=" << num_query_docs << ", k=" << k;

        // 3. Extract query features and aggregate
        auto feat_start = std::chrono::high_resolution_clock::now();
        std::vector<float> query_feats(num_query_docs * final_hidden_dim_, 0.0f);

        for (size_t q = 0; q < num_query_docs; ++q) {
            size_t q_vec_start = query_offset.offset[q];
            size_t q_vec_end = query_offset.offset[q + 1];
            size_t nq = q_vec_end - q_vec_start;

            // Extract features for each query token
            std::vector<float> token_feats(nq * final_hidden_dim_);
            mlp_->ExtractFeatures(query_data + q_vec_start * original_dim_, nq, token_feats.data());

            // Aggregate: sum (like LEMUR paper)
            float* q_feat = query_feats.data() + q * final_hidden_dim_;
            for (size_t t = 0; t < nq; ++t) {
                for (int32_t d = 0; d < final_hidden_dim_; ++d) {
                    q_feat[d] += token_feats[t * final_hidden_dim_ + d];
                }
            }

            // Normalize by number of tokens (optional, can use sum without normalization)
            // float inv_nq = 1.0f / nq;
            // for (int32_t d = 0; d < final_hidden_dim_; ++d) {
            //     q_feat[d] *= inv_nq;
            // }
        }
        auto feat_end = std::chrono::high_resolution_clock::now();
        double feat_ms = std::chrono::duration<double, std::milli>(feat_end - feat_start).count();

        LOG_KNOWHERE_INFO_ << "[LEMUR] Stage1 Feature extraction: " << feat_ms << " ms";

        // 4. ANN search on W
        bool do_rerank = config.lemur_rerank.value_or(true);
        auto query_feat_dataset = GenDataSet(num_query_docs, final_hidden_dim_, query_feats.data());

        int32_t ann_k;
        if (do_rerank) {
            auto retrieval_ann_ratio = config.retrieval_ann_ratio.value();
            ann_k = std::min(std::max((int32_t)(k * retrieval_ann_ratio), 1), (int32_t)num_docs_);
        } else {
            ann_k = std::min(k, (int32_t)num_docs_);
        }

        auto ann_start = std::chrono::high_resolution_clock::now();
        auto ann_result = ctx.ann_search(query_feat_dataset, ann_k);
        if (!ann_result.has_value()) {
            LOG_KNOWHERE_ERROR_ << "LEMUR ANN search failed: " << ann_result.what();
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "ANN search failed");
        }
        auto ann_end = std::chrono::high_resolution_clock::now();
        double ann_ms = std::chrono::duration<double, std::milli>(ann_end - ann_start).count();
        const auto* ann_ids = ann_result.value()->GetIds();

        LOG_KNOWHERE_INFO_ << "[LEMUR] Stage2 ANN search: " << ann_ms << " ms, ann_k=" << ann_k;

        // 5. If reranking disabled, return ANN results directly
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
            return GenResultDataSet((int64_t)num_query_docs, (int64_t)k, std::move(ids), std::move(dists));
        }

        // 6. MaxSim reranking (same as MUVERA)
        auto ids = std::make_unique<int64_t[]>(num_query_docs * k);
        auto dists = std::make_unique<float[]>(num_query_docs * k);

        auto rerank_start = std::chrono::high_resolution_clock::now();
        auto search_pool = ThreadPool::GetGlobalSearchThreadPool();

        // Statistics for logging
        size_t total_candidates = 0;
        size_t total_doc_vecs = 0;
        size_t total_query_vecs = 0;
        size_t total_distance_computations = 0;

        for (size_t q = 0; q < num_query_docs; ++q) {
            size_t q_vec_start = query_offset.offset[q];
            size_t q_vec_end = query_offset.offset[q + 1];
            size_t nq = q_vec_end - q_vec_start;
            total_query_vecs += nq;

            // Collect candidate doc IDs
            std::unordered_set<int64_t> candidate_docs;
            for (int32_t i = 0; i < ann_k; ++i) {
                int64_t doc_id = ann_ids[q * ann_k + i];
                if (doc_id >= 0 && doc_id < num_docs_) {
                    candidate_docs.insert(doc_id);
                }
            }
            total_candidates += candidate_docs.size();

            // Pre-allocate distance matrix and count doc vectors
            size_t max_doc_len = 0;
            size_t query_doc_vecs = 0;
            for (int64_t doc_id : candidate_docs) {
                size_t doc_len = emb_list_offset_->offset[doc_id + 1] - emb_list_offset_->offset[doc_id];
                max_doc_len = std::max(max_doc_len, doc_len);
                query_doc_vecs += doc_len;
            }
            total_doc_vecs += query_doc_vecs;
            total_distance_computations += nq * query_doc_vecs;
            std::vector<float> dist_matrix(nq * max_doc_len);

            auto compute_score = [&](int64_t doc_id) -> std::optional<float> {
                size_t doc_vec_start = emb_list_offset_->offset[doc_id];
                size_t doc_vec_end = emb_list_offset_->offset[doc_id + 1];
                size_t doc_len = doc_vec_end - doc_vec_start;

                const float* d_base = raw_data_.data() + doc_vec_start * original_dim_;

                std::vector<folly::Future<folly::Unit>> futs;
                futs.reserve(nq);

                for (size_t qi = 0; qi < nq; ++qi) {
                    futs.emplace_back(search_pool->push([&, qi = qi, doc_len = doc_len]() {
                        const float* q_vec = query_data + (q_vec_start + qi) * original_dim_;

                        if (sub_metric_type == metric::IP) {
                            faiss::fvec_inner_products_ny(dist_matrix.data() + qi * doc_len, q_vec, d_base,
                                                          original_dim_, doc_len);
                        } else if (sub_metric_type == metric::L2) {
                            faiss::fvec_L2sqr_ny(dist_matrix.data() + qi * doc_len, q_vec, d_base, original_dim_,
                                                 doc_len);
                        } else if (is_cosine) {
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
                            for (size_t di = 0; di < doc_len; ++di) {
                                const float* d_vec = d_base + di * original_dim_;
                                dist_matrix[qi * doc_len + di] =
                                    faiss::fvec_inner_product(q_vec, d_vec, original_dim_);
                            }
                        }
                    }));
                }

                WaitAllSuccess(futs);
                return get_sum_max_sim(dist_matrix.data(), nq, doc_len, larger_is_closer);
            };

            RerankCandidates(candidate_docs, k, larger_is_closer, compute_score, ids.get() + q * k,
                             dists.get() + q * k);
        }

        auto rerank_end = std::chrono::high_resolution_clock::now();
        double rerank_ms = std::chrono::duration<double, std::milli>(rerank_end - rerank_start).count();

        double avg_doc_len = total_candidates > 0 ? (double)total_doc_vecs / total_candidates : 0;
        double avg_query_len = num_query_docs > 0 ? (double)total_query_vecs / num_query_docs : 0;
        LOG_KNOWHERE_INFO_ << "[LEMUR] Stage3 Rerank: " << rerank_ms << " ms"
                           << ", total_candidates=" << total_candidates << ", avg_doc_len=" << avg_doc_len
                           << ", avg_query_len=" << avg_query_len
                           << ", total_dist_comps=" << total_distance_computations;

        return GenResultDataSet((int64_t)num_query_docs, (int64_t)k, std::move(ids), std::move(dists));
    }

    Status
    Serialize(BinarySet& binset) const override {
        // 1. Serialize config (including new fields: final_hidden_dim_, num_layers_)
        size_t config_size = 8 * sizeof(int32_t) + sizeof(int64_t) + 2 * sizeof(float);
        auto config_data = std::shared_ptr<uint8_t[]>(new uint8_t[config_size]);
        uint8_t* ptr = config_data.get();

        std::memcpy(ptr, &hidden_dim_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &final_hidden_dim_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_layers_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_train_samples_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_epochs_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &batch_size_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &seed_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &original_dim_, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(ptr, &num_docs_, sizeof(int64_t));
        ptr += sizeof(int64_t);
        std::memcpy(ptr, &label_mean_, sizeof(float));
        ptr += sizeof(float);
        std::memcpy(ptr, &label_std_, sizeof(float));

        binset.Append(meta::LEMUR_CONFIG, config_data, config_size);

        // 2. Serialize MLP weights (multi-layer)
        if (mlp_) {
            const auto& fc_weights = mlp_->GetFcWeights();
            const auto& fc_biases = mlp_->GetFcBiases();
            const auto& ln_gammas = mlp_->GetLnGammas();
            const auto& ln_betas = mlp_->GetLnBetas();
            const auto& W_out = mlp_->GetOutputWeights();

            // Calculate total size
            size_t mlp_size = sizeof(int32_t);  // num_layers
            for (int32_t i = 0; i < num_layers_; ++i) {
                mlp_size += 4 * sizeof(size_t);  // sizes for fc_w, fc_b, ln_g, ln_b
                mlp_size += fc_weights[i].size() * sizeof(float);
                mlp_size += fc_biases[i].size() * sizeof(float);
                mlp_size += ln_gammas[i].size() * sizeof(float);
                mlp_size += ln_betas[i].size() * sizeof(float);
            }
            mlp_size += sizeof(size_t) + W_out.size() * sizeof(float);  // output layer

            auto mlp_data = std::shared_ptr<uint8_t[]>(new uint8_t[mlp_size]);
            ptr = mlp_data.get();

            std::memcpy(ptr, &num_layers_, sizeof(int32_t));
            ptr += sizeof(int32_t);

            for (int32_t i = 0; i < num_layers_; ++i) {
                size_t sz = fc_weights[i].size();
                std::memcpy(ptr, &sz, sizeof(size_t));
                ptr += sizeof(size_t);
                std::memcpy(ptr, fc_weights[i].data(), sz * sizeof(float));
                ptr += sz * sizeof(float);

                sz = fc_biases[i].size();
                std::memcpy(ptr, &sz, sizeof(size_t));
                ptr += sizeof(size_t);
                std::memcpy(ptr, fc_biases[i].data(), sz * sizeof(float));
                ptr += sz * sizeof(float);

                sz = ln_gammas[i].size();
                std::memcpy(ptr, &sz, sizeof(size_t));
                ptr += sizeof(size_t);
                std::memcpy(ptr, ln_gammas[i].data(), sz * sizeof(float));
                ptr += sz * sizeof(float);

                sz = ln_betas[i].size();
                std::memcpy(ptr, &sz, sizeof(size_t));
                ptr += sizeof(size_t);
                std::memcpy(ptr, ln_betas[i].data(), sz * sizeof(float));
                ptr += sz * sizeof(float);
            }

            size_t sz = W_out.size();
            std::memcpy(ptr, &sz, sizeof(size_t));
            ptr += sizeof(size_t);
            std::memcpy(ptr, W_out.data(), sz * sizeof(float));

            binset.Append(meta::LEMUR_MLP, mlp_data, mlp_size);
        }

        // 3. Serialize document offsets
        if (emb_list_offset_) {
            size_t num_offsets = emb_list_offset_->offset.size();
            size_t offset_size = sizeof(size_t) + num_offsets * sizeof(size_t);
            auto offset_data = std::shared_ptr<uint8_t[]>(new uint8_t[offset_size]);

            std::memcpy(offset_data.get(), &num_offsets, sizeof(size_t));
            std::memcpy(offset_data.get() + sizeof(size_t), emb_list_offset_->offset.data(),
                        num_offsets * sizeof(size_t));

            binset.Append(meta::EMB_LIST_META, offset_data, offset_size);
        }

        // 4. Serialize raw data
        if (!raw_data_.empty()) {
            size_t raw_size = raw_data_.size() * sizeof(float);
            auto raw_data_bin = std::shared_ptr<uint8_t[]>(new uint8_t[raw_size]);
            std::memcpy(raw_data_bin.get(), raw_data_.data(), raw_size);

            binset.Append(meta::LEMUR_RAW_DATA, raw_data_bin, raw_size);
        }

        // 5. Serialize W matrix
        if (!W_.empty()) {
            size_t w_size = W_.size() * sizeof(float);
            auto w_data = std::shared_ptr<uint8_t[]>(new uint8_t[w_size]);
            std::memcpy(w_data.get(), W_.data(), w_size);

            binset.Append(meta::LEMUR_W, w_data, w_size);
        }

        LOG_KNOWHERE_INFO_ << "LEMUR Serialize completed";
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, const BaseConfig& config) override {
        // 1. Deserialize config
        auto config_bin = binset.GetByName(meta::LEMUR_CONFIG);
        if (!config_bin) {
            return Status::emb_list_inner_error;
        }

        const uint8_t* ptr = config_bin->data.get();
        std::memcpy(&hidden_dim_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&final_hidden_dim_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&num_layers_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&num_train_samples_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&num_epochs_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&batch_size_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&seed_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&original_dim_, ptr, sizeof(int32_t));
        ptr += sizeof(int32_t);
        std::memcpy(&num_docs_, ptr, sizeof(int64_t));
        ptr += sizeof(int64_t);
        std::memcpy(&label_mean_, ptr, sizeof(float));
        ptr += sizeof(float);
        std::memcpy(&label_std_, ptr, sizeof(float));

        LOG_KNOWHERE_INFO_ << "LEMUR Deserialize config: hidden_dim=" << hidden_dim_
                           << ", final_hidden_dim=" << final_hidden_dim_ << ", num_layers=" << num_layers_
                           << ", original_dim=" << original_dim_ << ", num_docs=" << num_docs_;

        // 2. Deserialize MLP weights (multi-layer)
        auto mlp_bin = binset.GetByName(meta::LEMUR_MLP);
        if (mlp_bin) {
            mlp_ = std::make_unique<SimpleMLP>(original_dim_, num_docs_, hidden_dim_, final_hidden_dim_, num_layers_,
                                               seed_);

            ptr = mlp_bin->data.get();

            int32_t saved_num_layers;
            std::memcpy(&saved_num_layers, ptr, sizeof(int32_t));
            ptr += sizeof(int32_t);

            std::vector<std::vector<float>> fc_weights(saved_num_layers);
            std::vector<std::vector<float>> fc_biases(saved_num_layers);
            std::vector<std::vector<float>> ln_gammas(saved_num_layers);
            std::vector<std::vector<float>> ln_betas(saved_num_layers);

            for (int32_t i = 0; i < saved_num_layers; ++i) {
                size_t sz;

                std::memcpy(&sz, ptr, sizeof(size_t));
                ptr += sizeof(size_t);
                fc_weights[i].resize(sz);
                std::memcpy(fc_weights[i].data(), ptr, sz * sizeof(float));
                ptr += sz * sizeof(float);

                std::memcpy(&sz, ptr, sizeof(size_t));
                ptr += sizeof(size_t);
                fc_biases[i].resize(sz);
                std::memcpy(fc_biases[i].data(), ptr, sz * sizeof(float));
                ptr += sz * sizeof(float);

                std::memcpy(&sz, ptr, sizeof(size_t));
                ptr += sizeof(size_t);
                ln_gammas[i].resize(sz);
                std::memcpy(ln_gammas[i].data(), ptr, sz * sizeof(float));
                ptr += sz * sizeof(float);

                std::memcpy(&sz, ptr, sizeof(size_t));
                ptr += sizeof(size_t);
                ln_betas[i].resize(sz);
                std::memcpy(ln_betas[i].data(), ptr, sz * sizeof(float));
                ptr += sz * sizeof(float);
            }

            size_t sz;
            std::memcpy(&sz, ptr, sizeof(size_t));
            ptr += sizeof(size_t);
            std::vector<float> W_out(sz);
            std::memcpy(W_out.data(), ptr, sz * sizeof(float));

            mlp_->SetFcWeights(fc_weights);
            mlp_->SetFcBiases(fc_biases);
            mlp_->SetLnGammas(ln_gammas);
            mlp_->SetLnBetas(ln_betas);
            mlp_->SetOutputWeights(W_out);
        }

        // 3. Deserialize document offsets
        auto offset_bin = binset.GetByName(meta::EMB_LIST_META);
        if (!offset_bin) {
            return Status::emb_list_inner_error;
        }

        size_t num_offsets = 0;
        std::memcpy(&num_offsets, offset_bin->data.get(), sizeof(size_t));
        std::vector<size_t> offset(num_offsets);
        std::memcpy(offset.data(), offset_bin->data.get() + sizeof(size_t), num_offsets * sizeof(size_t));
        emb_list_offset_ = std::make_shared<EmbListOffset>(std::move(offset));

        // 4. Deserialize raw data
        auto raw_bin = binset.GetByName(meta::LEMUR_RAW_DATA);
        if (!raw_bin) {
            return Status::emb_list_inner_error;
        }

        size_t num_floats = raw_bin->size / sizeof(float);
        raw_data_.resize(num_floats);
        std::memcpy(raw_data_.data(), raw_bin->data.get(), raw_bin->size);

        // 5. Deserialize W matrix
        auto w_bin = binset.GetByName(meta::LEMUR_W);
        if (w_bin) {
            size_t w_floats = w_bin->size / sizeof(float);
            W_.resize(w_floats);
            std::memcpy(W_.data(), w_bin->data.get(), w_bin->size);
        }

        LOG_KNOWHERE_INFO_ << "LEMUR Deserialize completed";
        return Status::success;
    }

    int32_t
    GetIndexedDim() const override {
        return final_hidden_dim_;
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
    // Config
    int32_t hidden_dim_ = 256;
    int32_t final_hidden_dim_ = 256;
    int32_t num_layers_ = 2;
    int32_t num_train_samples_ = 10000;
    int32_t num_epochs_ = 50;
    int32_t batch_size_ = 64;
    float learning_rate_ = 0.001f;
    int32_t seed_ = 42;

    int32_t original_dim_ = 0;
    int64_t num_docs_ = 0;

    // Label normalization
    float label_mean_ = 0.0f;
    float label_std_ = 1.0f;

    // MLP model
    std::unique_ptr<SimpleMLP> mlp_;

    // W matrix [num_docs, hidden_dim] - document representations
    std::vector<float> W_;

    // Storage for reranking
    std::shared_ptr<EmbListOffset> emb_list_offset_;
    std::vector<float> raw_data_;

    /**
     * @brief Compute MaxSim labels for training (BLAS optimized).
     *
     * For each sample vector v and each document d:
     *   label[v, d] = MaxSim(v, d) = max over all tokens t in d: IP(v, t)
     *
     * Optimization: Use cblas_sgemm to compute all inner products in batch,
     * then parallel reduce to MaxSim per document.
     */
    void
    ComputeMaxSimLabels(const float* samples, int32_t num_samples, const float* raw_data, const EmbListOffset& offset,
                        float* labels) const {
        openblas_set_num_threads(8);

        auto pool = ThreadPool::GetGlobalSearchThreadPool();
        size_t total_vectors = offset.offset.back();

        // Process samples in batches to limit memory usage
        // Larger batch = fewer sgemm calls = faster, but more memory
        // batch_size=1024: ~2.7GB per batch for 674K vectors
        const int32_t sample_batch_size = 2048;

        for (int32_t sample_start = 0; sample_start < num_samples; sample_start += sample_batch_size) {
            int32_t sample_end = std::min(sample_start + sample_batch_size, num_samples);
            int32_t batch_samples = sample_end - sample_start;

            // Compute all inner products: IP[batch_samples, total_vectors] = samples_batch @ raw_data.T
            std::vector<float> all_ips(batch_samples * total_vectors);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch_samples, total_vectors, original_dim_, 1.0f,
                        samples + sample_start * original_dim_, original_dim_, raw_data, original_dim_, 0.0f,
                        all_ips.data(), total_vectors);

            // Reduce to MaxSim per document (parallel over samples)
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(batch_samples);

            for (int32_t b = 0; b < batch_samples; ++b) {
                futs.emplace_back(pool->push([&, b, sample_start]() {
                    const float* ip_row = all_ips.data() + b * total_vectors;
                    float* label_row = labels + (sample_start + b) * num_docs_;

                    for (int64_t d = 0; d < num_docs_; ++d) {
                        size_t doc_start = offset.offset[d];
                        size_t doc_end = offset.offset[d + 1];
                        size_t doc_len = doc_end - doc_start;

                        // Find max similarity for this document
                        label_row[d] = FindMax(ip_row + doc_start, doc_len);
                    }
                }));
            }

            WaitAllSuccess(futs);
        }
    }
};

EmbListStrategyPtr
CreateLemurEmbListStrategy() {
    return std::make_unique<LemurEmbListStrategy>();
}

}  // namespace knowhere
