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

#ifndef EMB_LIST_STRATEGY_H
#define EMB_LIST_STRATEGY_H

#include <functional>
#include <memory>
#include <string>

#include "knowhere/binaryset.h"
#include "knowhere/bitsetview.h"
#include "knowhere/dataset.h"
#include "knowhere/emb_list_utils.h"
#include "knowhere/expected.h"

namespace knowhere {

class BaseConfig;

/**
 * @brief Simple iterator interface for incremental ANN result fetching.
 */
class AnnResultIterator {
 public:
    virtual ~AnnResultIterator() = default;
    virtual std::pair<int64_t, float> Next() = 0;
    virtual bool HasNext() = 0;
};
using AnnResultIteratorPtr = std::shared_ptr<AnnResultIterator>;

/**
 * @brief Context providing callbacks and resources for strategy search.
 *
 * Strategies have full control over search flow and can use these callbacks
 * as needed. Not all strategies use all callbacks.
 */
struct EmbListSearchContext {
    /**
     * @brief Execute ANN search on the underlying index.
     *
     * @param query Query dataset
     * @param k Number of results per query
     * @return Search results [nq, k] with ids and distances
     */
    std::function<expected<DataSetPtr>(const DataSetPtr query, int32_t k)> ann_search;

    /**
     * @brief Get ANN iterators for incremental result fetching.
     *
     * Returns one iterator per query vector. Use this when you need to collect
     * results incrementally (e.g., until enough unique docs are found).
     *
     * @param query Query dataset [nq, dim]
     * @return Vector of iterators, one per query
     */
    std::function<expected<std::vector<AnnResultIteratorPtr>>(const DataSetPtr query)> ann_iterator;

    /**
     * @brief Calculate distances between query vectors and indexed vectors by IDs.
     *
     * @param query Query vectors [nq, dim]
     * @param ids Vector IDs to compute distances for
     * @param ids_len Number of IDs
     * @param is_cosine Whether to use cosine similarity
     * @return Distance matrix [nq, ids_len]
     */
    std::function<expected<DataSetPtr>(const DataSetPtr query, const int64_t* ids, size_t ids_len, bool is_cosine)>
        calc_distance_by_ids;

    /**
     * @brief Retrieve raw vectors by their IDs.
     *
     * @param ids Vector IDs to retrieve
     * @param ids_len Number of IDs
     * @return Raw vectors [ids_len, dim]
     */
    std::function<expected<DataSetPtr>(const int64_t* ids, size_t ids_len)> get_vectors_by_ids;

    /**
     * @brief Get total count of indexed items.
     */
    std::function<int64_t()> get_index_count;

    /**
     * @brief Get query code size for the dataset.
     *
     * @param dataset Query dataset
     * @return Code size in bytes, or error
     */
    std::function<expected<size_t>(const DataSetPtr dataset)> get_query_code_size;

    /**
     * @brief Filtering bitset (document level for most strategies).
     */
    BitsetView bitset;
};

/**
 * @brief Abstract interface for EmbList encoding/search strategies.
 *
 * EmbList (Embedding List) represents multi-vector documents where each document
 * consists of multiple vectors. Different strategies handle these differently:
 *
 * - Direct: Index all vectors, search at vector level, aggregate to document level
 * - MUVERA: Encode each document to single vector (FDE), search encoded vectors, rerank with MaxSim
 * - PLAID: Use centroid-based retrieval with inverted index, then exact MaxSim
 *
 * Strategies have full control over the search flow and can implement arbitrary
 * multi-stage pipelines.
 */
class EmbListStrategy {
 public:
    virtual ~EmbListStrategy() = default;

    /**
     * @brief Strategy type identifier
     */
    virtual std::string
    Type() const = 0;

    /**
     * @brief Prepare data for building the underlying ANN index.
     *
     * @param dataset Original dataset containing all vectors [N, dim]
     * @param doc_offset Document offsets defining vector groupings
     * @param config Build configuration
     * @return Dataset to be indexed by underlying ANN index, or nullopt if no ANN index needed
     *
     * Direct: returns original dataset (index all N vectors)
     * MUVERA: returns FDE-encoded dataset [M, encoded_dim] where M = num_docs
     * PLAID: returns centroid vectors for ANN indexing
     */
    virtual expected<std::optional<DataSetPtr>>
    PrepareDataForBuild(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) = 0;

    /**
     * @brief Called after underlying ANN index is built.
     *
     * Allows strategy to perform post-build setup (e.g., storing raw data,
     * building additional indexes like inverted index for PLAID).
     *
     * @param dataset Original dataset
     * @param doc_offset Document offsets
     * @param config Build configuration
     * @return Status
     */
    virtual Status
    OnBuildComplete(const DataSetPtr dataset, const EmbListOffset& doc_offset, const BaseConfig& config) {
        return Status::success;
    }

    /**
     * @brief Check if strategy needs ID mapping for bitset filtering.
     *
     * Direct strategy needs vector_id -> doc_id mapping for 1-hop bitset check.
     * MUVERA/PLAID don't need this since they index at document/centroid level.
     */
    virtual bool
    NeedsBaseIndexIDMap() const = 0;

    /**
     * @brief Execute search with full control over the search flow.
     *
     * Strategy controls the entire search pipeline and can implement arbitrary
     * multi-stage retrieval (e.g., PLAID's centroid -> inverted index -> exact scoring).
     *
     * @param query_dataset Original query dataset containing query vectors
     * @param query_offset Query document offsets (queries can also be multi-vector)
     * @param k Number of documents to return per query
     * @param config Search configuration
     * @param ctx Search context providing callbacks (ANN search, distance calc, etc.)
     * @return Search results [num_query_docs, k] with document IDs and scores
     */
    virtual expected<DataSetPtr>
    Search(const DataSetPtr query_dataset, const EmbListOffset& query_offset, int32_t k, const BaseConfig& config,
           const EmbListSearchContext& ctx) const = 0;

    /**
     * @brief Serialize strategy-specific data.
     *
     * @param binset Binary set to append data to
     * @return Status
     */
    virtual Status
    Serialize(BinarySet& binset) const = 0;

    /**
     * @brief Deserialize strategy-specific data.
     *
     * @param binset Binary set containing serialized data
     * @param config Configuration for deserialization
     * @return Status
     */
    virtual Status
    Deserialize(const BinarySet& binset, const BaseConfig& config) = 0;

    /**
     * @brief Get emb_list offset structure (shared).
     *
     * Returns nullptr if strategy doesn't maintain emb_list offsets.
     */
    virtual std::shared_ptr<EmbListOffset>
    GetEmbListOffset() const {
        return nullptr;
    }

    /**
     * @brief Get the dimension of vectors indexed by underlying ANN.
     *
     * Direct: original dim
     * MUVERA: encoded dim
     * PLAID: centroid dim (same as original)
     *
     * Returns 0 if strategy doesn't use ANN index.
     */
    virtual int32_t
    GetIndexedDim() const = 0;

    /**
     * @brief Get number of documents.
     */
    virtual int64_t
    GetDocCount() const = 0;
};

using EmbListStrategyPtr = std::unique_ptr<EmbListStrategy>;

/**
 * @brief Factory function to create EmbList strategy.
 *
 * @param strategy_type Strategy type: "direct", "muvera", "plaid", etc.
 * @param config Configuration containing strategy-specific parameters
 * @return Strategy instance or error
 */
expected<EmbListStrategyPtr>
CreateEmbListStrategy(const std::string& strategy_type, const BaseConfig& config);

}  // namespace knowhere

#endif /* EMB_LIST_STRATEGY_H */
