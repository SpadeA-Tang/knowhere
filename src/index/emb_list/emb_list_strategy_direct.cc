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
#include <functional>
#include <limits>
#include <queue>
#include <unordered_set>

#include "knowhere/comp/index_param.h"
#include "knowhere/config.h"
#include "knowhere/index/emb_list_strategy.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

// DirectEmbListStrategy indexes all vectors and aggregates scores at search time.
class DirectEmbListStrategy : public EmbListStrategy {
 public:
    std::string
    Type() const override {
        return "direct";
    }

    expected<std::optional<DataSetPtr>>
    PrepareDataForBuild(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) override {
        emb_list_offset_ = std::make_shared<EmbListOffset>(doc_offset.offset);
        original_dim_ = dataset->GetDim();
        return std::optional<DataSetPtr>(dataset);
    }

    bool
    NeedsBaseIndexIDMap() const override {
        return true;  // needs vector_id -> doc_id mapping for bitset filtering
    }

    expected<DataSetPtr>
    Search(const DataSetPtr query_dataset, const EmbListOffset& query_offset, int32_t k, const BaseConfig& config,
           const EmbListSearchContext& ctx) const override {
        if (!emb_list_offset_) {
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "emb_list_offset not initialized");
        }

        auto metric_type = config.metric_type.value();
        auto el_metric_type_or = get_el_metric_type(metric_type);
        if (!el_metric_type_or.has_value()) {
            LOG_KNOWHERE_WARNING_ << "Invalid metric type: " << metric_type;
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "invalid metric type");
        }
        auto el_metric_type = el_metric_type_or.value();
        LOG_KNOWHERE_DEBUG_ << "search emb_list with el metric_type: " << el_metric_type;
        auto el_agg_func_or = get_emb_list_agg_func(el_metric_type);
        if (!el_agg_func_or.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid emb list aggeration function for metric type: " << el_metric_type;
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error,
                                             "invalid emb list aggeration function for metric type: " + el_metric_type);
        }
        auto el_agg_func = el_agg_func_or.value();

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
        bool is_cosine = sub_metric_type == metric::COSINE ? true : false;
        LOG_KNOWHERE_DEBUG_ << "search emb_list with sub metric_type: " << sub_metric_type;

        auto dim = query_dataset->GetDim();
        auto num_q_el = query_offset.num_el();

        auto query_code_size_or = ctx.get_query_code_size(query_dataset);
        if (!query_code_size_or.has_value()) {
            LOG_KNOWHERE_ERROR_ << "could not get query code size";
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "could not get query code size");
        }
        auto query_code_size = query_code_size_or.value();

        // Stage 1: Use iterators to collect enough unique docs
        auto retrieval_ann_ratio = config.retrieval_ann_ratio.value();
        if (retrieval_ann_ratio <= 0.0f) {
            auto err_msg = "retrieval_ann_ratio could not be less than or equal to 0";
            LOG_KNOWHERE_WARNING_ << err_msg;
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, err_msg);
        }
        // Target number of unique docs to collect per query doc
        size_t min_unique_docs = std::min(std::max((size_t)(k * retrieval_ann_ratio), (size_t)1),
                                          (size_t)emb_list_offset_->num_el());

        auto stage1_start = std::chrono::high_resolution_clock::now();

        // Get iterators for all query vectors
        auto iterators_result = ctx.ann_iterator(query_dataset);
        if (!iterators_result.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to get ANN iterators: " << iterators_result.what();
            return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "failed to get ANN iterators");
        }
        auto& iterators = iterators_result.value();

        auto stage1_end = std::chrono::high_resolution_clock::now();
        double stage1_ms = std::chrono::duration<double, std::milli>(stage1_end - stage1_start).count();
        auto total_query_vecs = query_dataset->GetRows();
        LOG_KNOWHERE_INFO_ << "[Direct] Stage1 Iterator init: " << stage1_ms << " ms"
                           << ", num_query_docs=" << num_q_el << ", num_query_vecs=" << total_query_vecs
                           << ", avg_vecs_per_query=" << (num_q_el > 0 ? (double)total_query_vecs / num_q_el : 0)
                           << ", k=" << k << ", retrieval_ann_ratio=" << retrieval_ann_ratio
                           << ", min_unique_docs=" << min_unique_docs
                           << ", index_docs=" << emb_list_offset_->num_el()
                           << ", index_vecs=" << (emb_list_offset_->offset.back());

        auto ids = std::make_unique<int64_t[]>(num_q_el * k);
        auto dists = std::make_unique<float[]>(num_q_el * k);

        // Stage 2: For each query doc, collect unique docs via iterators, then aggregate scores
        auto stage2_start = std::chrono::high_resolution_clock::now();
        double total_candidate_collect_ms = 0;
        double total_rerank_ms = 0;
        size_t total_candidates = 0;
        size_t total_distance_computations = 0;
        size_t total_vecs_traversed = 0;

        for (size_t i = 0; i < num_q_el; i++) {
            auto start_offset = query_offset.offset[i];
            auto end_offset = query_offset.offset[i + 1];
            auto nq = end_offset - start_offset;

            auto collect_start = std::chrono::high_resolution_clock::now();

            // Collect unique doc IDs using iterators for this query doc's vectors
            std::unordered_set<size_t> el_ids_set;

            // Use a min-heap to merge results from multiple iterators by distance
            // pair: (distance, (iterator_index, vec_id))
            using HeapItem = std::pair<float, std::pair<size_t, int64_t>>;
            using HeapCmp = std::function<bool(const HeapItem&, const HeapItem&)>;
            HeapCmp cmp = larger_is_closer
                              ? HeapCmp([](const HeapItem& a, const HeapItem& b) { return a.first < b.first; })
                              : HeapCmp([](const HeapItem& a, const HeapItem& b) { return a.first > b.first; });
            std::priority_queue<HeapItem, std::vector<HeapItem>, HeapCmp> heap(cmp);

            // Initialize heap with first result from each iterator (one per query vector)
            size_t has_next_count = 0;
            size_t valid_vec_count = 0;
            for (size_t j = start_offset; j < end_offset; j++) {
                if (iterators[j]->HasNext()) {
                    has_next_count++;
                    auto [vec_id, dist] = iterators[j]->Next();
                    if (vec_id >= 0) {
                        valid_vec_count++;
                        heap.push({dist, {j, vec_id}});
                    }
                }
            }
            LOG_KNOWHERE_DEBUG_ << "[Direct] Query doc " << i << ": start=" << start_offset << ", end=" << end_offset
                                << ", has_next_count=" << has_next_count << ", valid_vec_count=" << valid_vec_count
                                << ", heap_size=" << heap.size();

            // Collect results until we have enough unique docs
            size_t vecs_traversed = 0;
            while (!heap.empty() && el_ids_set.size() < min_unique_docs) {
                auto [dist, iter_info] = heap.top();
                heap.pop();
                auto [iter_idx, vec_id] = iter_info;
                vecs_traversed++;

                // Map vec_id to doc_id
                size_t doc_id = emb_list_offset_->get_el_id((size_t)vec_id);
                el_ids_set.insert(doc_id);

                // Get next result from this iterator
                if (iterators[iter_idx]->HasNext()) {
                    auto [next_vec_id, next_dist] = iterators[iter_idx]->Next();
                    if (next_vec_id >= 0) {
                        heap.push({next_dist, {iter_idx, next_vec_id}});
                    }
                }
            }
            total_vecs_traversed += vecs_traversed;

            // Convert to int64_t set for RerankCandidates
            std::unordered_set<int64_t> candidate_docs;
            for (const auto& el_id : el_ids_set) {
                if (el_id < emb_list_offset_->num_el()) {
                    candidate_docs.insert((int64_t)el_id);
                }
            }
            auto collect_end = std::chrono::high_resolution_clock::now();
            total_candidate_collect_ms +=
                std::chrono::duration<double, std::milli>(collect_end - collect_start).count();
            total_candidates += candidate_docs.size();

            // Compute MaxSim score for each candidate
            auto tensor = (const char*)query_dataset->GetTensor();
            size_t tensor_offset = start_offset * query_code_size;
            auto bf_query_dataset = GenDataSet(nq, dim, tensor + tensor_offset);

            bool has_error = false;
            std::string error_msg;
            auto rerank_start = std::chrono::high_resolution_clock::now();
            auto compute_score = [&](int64_t doc_id) -> std::optional<float> {
                if (has_error) {
                    return std::nullopt;
                }

                auto vids = emb_list_offset_->get_vids((size_t)doc_id);
                total_distance_computations += nq * vids.size();
                auto bf_search_res = ctx.calc_distance_by_ids(bf_query_dataset, vids.data(), vids.size(), is_cosine);
                if (!bf_search_res.has_value()) {
                    has_error = true;
                    error_msg = bf_search_res.what();
                    return std::nullopt;
                }
                const auto* bf_dists = bf_search_res.value()->GetDistance();
                return el_agg_func(bf_dists, nq, vids.size(), larger_is_closer);
            };

            RerankCandidates(candidate_docs, k, larger_is_closer, compute_score, ids.get() + i * k,
                             dists.get() + i * k);
            auto rerank_end = std::chrono::high_resolution_clock::now();
            total_rerank_ms += std::chrono::duration<double, std::milli>(rerank_end - rerank_start).count();

            if (has_error) {
                LOG_KNOWHERE_ERROR_ << "bf search error: " << error_msg;
                return expected<DataSetPtr>::Err(Status::emb_list_inner_error, "bf search error");
            }
        }

        auto stage2_end = std::chrono::high_resolution_clock::now();
        double stage2_ms = std::chrono::duration<double, std::milli>(stage2_end - stage2_start).count();
        LOG_KNOWHERE_INFO_ << "[Direct] Stage2 Rerank: " << stage2_ms << " ms"
                           << ", candidate_collect=" << total_candidate_collect_ms << " ms"
                           << ", rerank_compute=" << total_rerank_ms << " ms"
                           << ", total_vecs_traversed=" << total_vecs_traversed
                           << ", avg_vecs_traversed_per_query=" << (num_q_el > 0 ? (double)total_vecs_traversed / num_q_el : 0)
                           << ", total_candidates=" << total_candidates
                           << ", avg_candidates_per_query=" << (num_q_el > 0 ? (double)total_candidates / num_q_el : 0)
                           << ", total_dist_comps=" << total_distance_computations
                           << ", avg_dist_comps_per_query=" << (num_q_el > 0 ? (double)total_distance_computations / num_q_el : 0);

        return GenResultDataSet((int64_t)num_q_el, (int64_t)k, std::move(ids), std::move(dists));
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (!emb_list_offset_) {
            return Status::success;
        }
        size_t num_offsets = emb_list_offset_->offset.size();
        size_t total_bytes = sizeof(size_t) + num_offsets * sizeof(size_t);
        auto data = std::shared_ptr<uint8_t[]>(new uint8_t[total_bytes]);
        std::memcpy(data.get(), &num_offsets, sizeof(size_t));
        std::memcpy(data.get() + sizeof(size_t), emb_list_offset_->offset.data(), num_offsets * sizeof(size_t));
        binset.Append(meta::EMB_LIST_META, data, total_bytes);
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, const BaseConfig& config) override {
        auto binary = binset.GetByName(meta::EMB_LIST_META);
        if (!binary) {
            LOG_KNOWHERE_WARNING_ << "No emb_list offset found in binary set";
            return Status::emb_list_inner_error;
        }

        size_t num_offsets = 0;
        std::memcpy(&num_offsets, binary->data.get(), sizeof(size_t));
        std::vector<size_t> offset(num_offsets);
        std::memcpy(offset.data(), binary->data.get() + sizeof(size_t), num_offsets * sizeof(size_t));
        emb_list_offset_ = std::make_shared<EmbListOffset>(std::move(offset));
        return Status::success;
    }

    std::shared_ptr<EmbListOffset>
    GetEmbListOffset() const override {
        return emb_list_offset_;
    }

    int32_t
    GetIndexedDim() const override {
        return original_dim_;
    }

    int64_t
    GetDocCount() const override {
        return emb_list_offset_ ? emb_list_offset_->num_el() : 0;
    }

 private:
    std::shared_ptr<EmbListOffset> emb_list_offset_;
    int32_t original_dim_ = 0;
};

EmbListStrategyPtr
CreateDirectEmbListStrategy() {
    return std::make_unique<DirectEmbListStrategy>();
}

}  // namespace knowhere
