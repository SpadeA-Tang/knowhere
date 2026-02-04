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

#ifndef SIMPLE_MLP_H
#define SIMPLE_MLP_H

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "knowhere/log.h"

// OpenBLAS thread control (defined in OpenBLAS library)
extern "C" {
void openblas_set_num_threads(int num_threads);
int openblas_get_num_threads(void);
}

namespace knowhere {


/**
 * @brief MLP implementation matching LEMUR paper/github.
 *
 * Architecture (num_layers=2, default):
 *   input -> [Linear -> LayerNorm -> GELU] -> [Linear -> LayerNorm -> GELU] -> Linear(no bias) -> output
 *            |<-------------- feature_extractor ---------------->|            |<-- output_layer -->|
 *
 * For LEMUR:
 *   - feature_extractor: extracts hidden features for query encoding
 *   - output_layer weights (W2): document representations [num_docs, final_hidden_dim]
 */
class SimpleMLP {
 public:
    /**
     * @brief Construct MLP matching LEMUR architecture.
     *
     * @param input_dim Input dimension (e.g., 128 for ColBERT)
     * @param output_dim Output dimension (num_docs)
     * @param hidden_dim Hidden layer dimension (default 1024)
     * @param final_hidden_dim Final hidden dimension before output (default same as hidden_dim)
     * @param num_layers Number of layers in feature_extractor (default 2)
     * @param seed Random seed for weight initialization
     */
    SimpleMLP(int32_t input_dim, int32_t output_dim, int32_t hidden_dim = 1024, int32_t final_hidden_dim = 0,
              int32_t num_layers = 2, int32_t seed = 42)
        : input_dim_(input_dim), output_dim_(output_dim), hidden_dim_(hidden_dim), num_layers_(num_layers) {
        if (final_hidden_dim <= 0) {
            final_hidden_dim_ = hidden_dim;
        } else {
            final_hidden_dim_ = final_hidden_dim;
        }

        std::mt19937 rng(seed);

        // Build feature_extractor layers
        // dims: [input_dim, hidden_dim, hidden_dim, ..., final_hidden_dim]
        std::vector<int32_t> dims;
        dims.push_back(input_dim);
        for (int32_t i = 0; i < num_layers_ - 1; ++i) {
            dims.push_back(hidden_dim_);
        }

        // Initialize weights for each layer in feature_extractor
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            int32_t in_dim = dims[layer];
            int32_t out_dim = (layer == num_layers_ - 1) ? final_hidden_dim_ : hidden_dim_;

            // Linear weights: [out_dim, in_dim]
            float scale = std::sqrt(2.0f / (in_dim + out_dim));
            std::normal_distribution<float> dist(0.0f, scale);

            std::vector<float> W(out_dim * in_dim);
            for (auto& w : W) {
                w = dist(rng);
            }
            std::vector<float> b(out_dim, 0.0f);

            fc_weights_.push_back(std::move(W));
            fc_biases_.push_back(std::move(b));

            // LayerNorm parameters: gamma (scale) = 1, beta (shift) = 0
            std::vector<float> gamma(out_dim, 1.0f);
            std::vector<float> beta(out_dim, 0.0f);

            ln_gammas_.push_back(std::move(gamma));
            ln_betas_.push_back(std::move(beta));

            layer_dims_.push_back(out_dim);
        }

        // Output layer: [output_dim, final_hidden_dim], no bias (matching LEMUR)
        float scale_out = std::sqrt(2.0f / (final_hidden_dim_ + output_dim_));
        std::normal_distribution<float> dist_out(0.0f, scale_out);
        W_out_.resize(output_dim_ * final_hidden_dim_);
        for (auto& w : W_out_) {
            w = dist_out(rng);
        }
        // No bias for output layer (LEMUR style)

        // Initialize gradient storage
        InitGradients();

        // Initialize Adam optimizer states
        InitAdamStates();
    }

    /**
     * @brief Forward pass through entire network.
     *
     * @param input Input matrix [batch_size, input_dim]
     * @param batch_size Number of samples
     * @param output Output matrix [batch_size, output_dim]
     * @param store_intermediates If true, store intermediate activations for backward pass
     */
    void
    Forward(const float* input, int32_t batch_size, float* output, bool store_intermediates = false) {
        if (store_intermediates) {
            intermediates_.clear();
            pre_ln_values_.clear();
            post_ln_values_.clear();
            // Store input
            intermediates_.emplace_back(input, input + batch_size * input_dim_);
        }

        // Current activation
        std::vector<float> current(input, input + batch_size * input_dim_);
        int32_t current_dim = input_dim_;

        // Forward through feature_extractor layers
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            int32_t out_dim = layer_dims_[layer];
            std::vector<float> linear_out(batch_size * out_dim);
            std::vector<float> ln_out(batch_size * out_dim);
            std::vector<float> act_out(batch_size * out_dim);

            // Linear: out = input @ W.T + b
            LinearForward(current.data(), fc_weights_[layer].data(), fc_biases_[layer].data(), batch_size, current_dim,
                          out_dim, linear_out.data());

            if (store_intermediates) {
                pre_ln_values_.push_back(linear_out);
            }

            // LayerNorm
            LayerNormForward(linear_out.data(), ln_gammas_[layer].data(), ln_betas_[layer].data(), batch_size, out_dim,
                             ln_out.data());

            if (store_intermediates) {
                post_ln_values_.push_back(ln_out);  // Store LayerNorm output for GELU backward
            }

            // GELU activation
            GELUForward(ln_out.data(), batch_size * out_dim, act_out.data());

            if (store_intermediates) {
                intermediates_.push_back(act_out);
            }

            current = std::move(act_out);
            current_dim = out_dim;
        }

        // Store final hidden for feature extraction
        final_hidden_ = current;

        // Output layer: out = hidden @ W_out.T (no bias)
        LinearForwardNoBias(current.data(), W_out_.data(), batch_size, final_hidden_dim_, output_dim_, output);
    }

    /**
     * @brief Extract features (output of feature_extractor, before output_layer).
     *
     * @param input Input matrix [batch_size, input_dim]
     * @param batch_size Number of samples
     * @param features Output features [batch_size, final_hidden_dim]
     */
    void
    ExtractFeatures(const float* input, int32_t batch_size, float* features) {
        std::vector<float> current(input, input + batch_size * input_dim_);
        int32_t current_dim = input_dim_;

        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            int32_t out_dim = layer_dims_[layer];
            std::vector<float> linear_out(batch_size * out_dim);
            std::vector<float> ln_out(batch_size * out_dim);
            std::vector<float> act_out(batch_size * out_dim);

            LinearForward(current.data(), fc_weights_[layer].data(), fc_biases_[layer].data(), batch_size, current_dim,
                          out_dim, linear_out.data());

            LayerNormForward(linear_out.data(), ln_gammas_[layer].data(), ln_betas_[layer].data(), batch_size, out_dim,
                             ln_out.data());

            GELUForward(ln_out.data(), batch_size * out_dim, act_out.data());

            current = std::move(act_out);
            current_dim = out_dim;
        }

        std::memcpy(features, current.data(), batch_size * final_hidden_dim_ * sizeof(float));
    }

    /**
     * @brief Backward pass and compute gradients.
     *
     * @param input Input matrix [batch_size, input_dim]
     * @param target Target values [batch_size, output_dim]
     * @param output Predicted values [batch_size, output_dim]
     * @param batch_size Number of samples
     * @return MSE loss
     */
    float
    Backward(const float* input, const float* target, const float* output, int32_t batch_size) {
        // Clear gradients
        ClearGradients();

        float total_loss = 0.0f;
        float scale = 1.0f / batch_size;

        // Compute output gradient: d_output = 2 * (output - target) / output_dim
        std::vector<float> d_output(batch_size * output_dim_);
        for (int32_t b = 0; b < batch_size; ++b) {
            for (int32_t j = 0; j < output_dim_; ++j) {
                float diff = output[b * output_dim_ + j] - target[b * output_dim_ + j];
                total_loss += diff * diff;
                d_output[b * output_dim_ + j] = 2.0f * diff * scale / output_dim_;
            }
        }

        // Backward through output layer (no bias)
        std::vector<float> d_hidden(batch_size * final_hidden_dim_, 0.0f);
        LinearBackwardNoBias(final_hidden_.data(), d_output.data(), batch_size, final_hidden_dim_, output_dim_,
                             dW_out_.data(), d_hidden.data());

        // Backward through feature_extractor layers (reverse order)
        std::vector<float> d_current = std::move(d_hidden);

        for (int32_t layer = num_layers_ - 1; layer >= 0; --layer) {
            int32_t out_dim = layer_dims_[layer];
            int32_t in_dim = (layer == 0) ? input_dim_ : layer_dims_[layer - 1];

            // Get stored intermediate values
            const float* layer_input = intermediates_[layer].data();
            const float* pre_ln = pre_ln_values_[layer].data();    // Linear output (before LayerNorm)
            const float* post_ln = post_ln_values_[layer].data();  // LayerNorm output (GELU input)

            std::vector<float> d_act(batch_size * out_dim);
            std::vector<float> d_ln(batch_size * out_dim);
            std::vector<float> d_input_layer(batch_size * in_dim, 0.0f);

            // Backward GELU - use post_ln (LayerNorm output) as GELU's input
            GELUBackward(post_ln, d_current.data(), batch_size * out_dim, d_act.data());

            // Backward LayerNorm
            LayerNormBackward(pre_ln, d_act.data(), ln_gammas_[layer].data(), batch_size, out_dim, d_ln.data(),
                              d_ln_gammas_[layer].data(), d_ln_betas_[layer].data());

            // Backward Linear
            LinearBackward(layer_input, d_ln.data(), fc_weights_[layer].data(), batch_size, in_dim, out_dim,
                           d_fc_weights_[layer].data(), d_fc_biases_[layer].data(), d_input_layer.data());

            d_current = std::move(d_input_layer);
        }

        return total_loss / (batch_size * output_dim_);
    }

    /**
     * @brief Update weights using Adam optimizer.
     */
    void
    UpdateAdam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        adam_t_++;
        float bc1 = 1.0f - std::pow(beta1, adam_t_);
        float bc2 = 1.0f - std::pow(beta2, adam_t_);

        // Update feature_extractor layers
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            AdamUpdate(fc_weights_[layer], d_fc_weights_[layer], m_fc_weights_[layer], v_fc_weights_[layer], lr, beta1,
                       beta2, eps, bc1, bc2);
            AdamUpdate(fc_biases_[layer], d_fc_biases_[layer], m_fc_biases_[layer], v_fc_biases_[layer], lr, beta1,
                       beta2, eps, bc1, bc2);
            AdamUpdate(ln_gammas_[layer], d_ln_gammas_[layer], m_ln_gammas_[layer], v_ln_gammas_[layer], lr, beta1,
                       beta2, eps, bc1, bc2);
            AdamUpdate(ln_betas_[layer], d_ln_betas_[layer], m_ln_betas_[layer], v_ln_betas_[layer], lr, beta1, beta2,
                       eps, bc1, bc2);
        }

        // Update output layer
        AdamUpdate(W_out_, dW_out_, m_W_out_, v_W_out_, lr, beta1, beta2, eps, bc1, bc2);
    }

    /**
     * @brief Train on a dataset.
     *
     * @param log_interval Log loss every log_interval epochs (0 = no logging except first/last)
     */
    float
    Train(const float* X_train, const float* y_train, int32_t num_samples, int32_t epochs = 100,
          int32_t batch_size = 64, float lr = 0.001f, bool verbose = false, int32_t log_interval = 3) {
        // Configure OpenBLAS threads based on matrix size
        // Too many threads cause synchronization overhead (red in htop = kernel mode)
        // Sweet spot is typically 4-8 threads for medium matrices
        int hw_threads = std::thread::hardware_concurrency();
        if (hw_threads < 1) {
            hw_threads = 4;
        }
        // Cap at 8 threads to avoid excessive synchronization overhead
        int num_threads = std::min(hw_threads, 8);
        openblas_set_num_threads(num_threads);

        std::vector<int32_t> indices(num_samples);
        for (int32_t i = 0; i < num_samples; ++i) {
            indices[i] = i;
        }

        std::vector<float> batch_input(batch_size * input_dim_);
        std::vector<float> batch_target(batch_size * output_dim_);
        std::vector<float> batch_output(batch_size * output_dim_);

        std::mt19937 rng(42);
        float final_loss = 0.0f;

        LOG_KNOWHERE_INFO_ << "[LEMUR MLP] Training started: samples=" << num_samples << ", epochs=" << epochs
                           << ", batch_size=" << batch_size << ", lr=" << lr
                           << ", openblas_threads=" << openblas_get_num_threads();

        // Early stopping parameters
        float best_loss = std::numeric_limits<float>::max();
        int32_t patience_counter = 0;
        const int32_t patience = 5;           // Stop if no improvement for 5 epochs
        const float min_delta = 1e-4f;        // Minimum improvement to reset patience

        for (int32_t epoch = 0; epoch < epochs; ++epoch) {
            std::shuffle(indices.begin(), indices.end(), rng);

            float epoch_loss = 0.0f;
            int32_t num_batches = 0;

            for (int32_t start = 0; start < num_samples; start += batch_size) {
                int32_t actual_batch = std::min(batch_size, num_samples - start);

                // Gather batch
                for (int32_t b = 0; b < actual_batch; ++b) {
                    int32_t idx = indices[start + b];
                    std::memcpy(batch_input.data() + b * input_dim_, X_train + idx * input_dim_,
                                input_dim_ * sizeof(float));
                    std::memcpy(batch_target.data() + b * output_dim_, y_train + idx * output_dim_,
                                output_dim_ * sizeof(float));
                }

                // Forward (store intermediates for backward)
                Forward(batch_input.data(), actual_batch, batch_output.data(), true);

                // Backward
                float loss = Backward(batch_input.data(), batch_target.data(), batch_output.data(), actual_batch);
                epoch_loss += loss;
                num_batches++;

                // Update
                UpdateAdam(lr);
            }

            final_loss = epoch_loss / num_batches;

            // Log training progress
            bool should_log =
                (epoch == 0) || (epoch == epochs - 1) || (log_interval > 0 && (epoch + 1) % log_interval == 0);
            if (should_log) {
                LOG_KNOWHERE_INFO_ << "[LEMUR MLP] Epoch " << (epoch + 1) << "/" << epochs << ", Loss: " << final_loss;
            }

            // Early stopping check
            if (best_loss - final_loss > min_delta) {
                best_loss = final_loss;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= patience) {
                    LOG_KNOWHERE_INFO_ << "[LEMUR MLP] Early stopping at epoch " << (epoch + 1)
                                       << ", loss=" << final_loss;
                    break;
                }
            }
        }

        LOG_KNOWHERE_INFO_ << "[LEMUR MLP] Training completed, final loss: " << final_loss;

        return final_loss;
    }

    // Getters
    const std::vector<float>&
    GetOutputWeights() const {
        return W_out_;
    }
    int32_t
    FinalHiddenDim() const {
        return final_hidden_dim_;
    }
    int32_t
    InputDim() const {
        return input_dim_;
    }
    int32_t
    OutputDim() const {
        return output_dim_;
    }
    int32_t
    NumLayers() const {
        return num_layers_;
    }
    int32_t
    HiddenDim() const {
        return hidden_dim_;
    }

    // For serialization
    const std::vector<std::vector<float>>&
    GetFcWeights() const {
        return fc_weights_;
    }
    const std::vector<std::vector<float>>&
    GetFcBiases() const {
        return fc_biases_;
    }
    const std::vector<std::vector<float>>&
    GetLnGammas() const {
        return ln_gammas_;
    }
    const std::vector<std::vector<float>>&
    GetLnBetas() const {
        return ln_betas_;
    }

    void
    SetFcWeights(const std::vector<std::vector<float>>& w) {
        fc_weights_ = w;
    }
    void
    SetFcBiases(const std::vector<std::vector<float>>& b) {
        fc_biases_ = b;
    }
    void
    SetLnGammas(const std::vector<std::vector<float>>& g) {
        ln_gammas_ = g;
    }
    void
    SetLnBetas(const std::vector<std::vector<float>>& b) {
        ln_betas_ = b;
    }
    void
    SetOutputWeights(const std::vector<float>& w) {
        W_out_ = w;
    }

 private:
    int32_t input_dim_;
    int32_t output_dim_;
    int32_t hidden_dim_;
    int32_t final_hidden_dim_;
    int32_t num_layers_;

    // Feature extractor layers
    std::vector<std::vector<float>> fc_weights_;  // [num_layers][out_dim * in_dim]
    std::vector<std::vector<float>> fc_biases_;   // [num_layers][out_dim]
    std::vector<std::vector<float>> ln_gammas_;   // [num_layers][out_dim]
    std::vector<std::vector<float>> ln_betas_;    // [num_layers][out_dim]
    std::vector<int32_t> layer_dims_;             // output dim of each layer

    // Output layer (no bias, matching LEMUR)
    std::vector<float> W_out_;  // [output_dim * final_hidden_dim]

    // Gradients
    std::vector<std::vector<float>> d_fc_weights_;
    std::vector<std::vector<float>> d_fc_biases_;
    std::vector<std::vector<float>> d_ln_gammas_;
    std::vector<std::vector<float>> d_ln_betas_;
    std::vector<float> dW_out_;

    // Adam states
    std::vector<std::vector<float>> m_fc_weights_, v_fc_weights_;
    std::vector<std::vector<float>> m_fc_biases_, v_fc_biases_;
    std::vector<std::vector<float>> m_ln_gammas_, v_ln_gammas_;
    std::vector<std::vector<float>> m_ln_betas_, v_ln_betas_;
    std::vector<float> m_W_out_, v_W_out_;
    int32_t adam_t_ = 0;

    // Intermediate values for backward pass
    std::vector<std::vector<float>> intermediates_;   // activations after each layer (input, act_out_0, act_out_1, ...)
    std::vector<std::vector<float>> pre_ln_values_;   // Linear outputs, before LayerNorm (for LN backward)
    std::vector<std::vector<float>> post_ln_values_;  // LayerNorm outputs, before GELU (for GELU backward)
    std::vector<float> final_hidden_;                 // output of feature_extractor

    void
    InitGradients() {
        d_fc_weights_.resize(num_layers_);
        d_fc_biases_.resize(num_layers_);
        d_ln_gammas_.resize(num_layers_);
        d_ln_betas_.resize(num_layers_);

        for (int32_t i = 0; i < num_layers_; ++i) {
            d_fc_weights_[i].resize(fc_weights_[i].size(), 0.0f);
            d_fc_biases_[i].resize(fc_biases_[i].size(), 0.0f);
            d_ln_gammas_[i].resize(ln_gammas_[i].size(), 0.0f);
            d_ln_betas_[i].resize(ln_betas_[i].size(), 0.0f);
        }

        dW_out_.resize(W_out_.size(), 0.0f);
    }

    void
    InitAdamStates() {
        m_fc_weights_.resize(num_layers_);
        v_fc_weights_.resize(num_layers_);
        m_fc_biases_.resize(num_layers_);
        v_fc_biases_.resize(num_layers_);
        m_ln_gammas_.resize(num_layers_);
        v_ln_gammas_.resize(num_layers_);
        m_ln_betas_.resize(num_layers_);
        v_ln_betas_.resize(num_layers_);

        for (int32_t i = 0; i < num_layers_; ++i) {
            m_fc_weights_[i].resize(fc_weights_[i].size(), 0.0f);
            v_fc_weights_[i].resize(fc_weights_[i].size(), 0.0f);
            m_fc_biases_[i].resize(fc_biases_[i].size(), 0.0f);
            v_fc_biases_[i].resize(fc_biases_[i].size(), 0.0f);
            m_ln_gammas_[i].resize(ln_gammas_[i].size(), 0.0f);
            v_ln_gammas_[i].resize(ln_gammas_[i].size(), 0.0f);
            m_ln_betas_[i].resize(ln_betas_[i].size(), 0.0f);
            v_ln_betas_[i].resize(ln_betas_[i].size(), 0.0f);
        }

        m_W_out_.resize(W_out_.size(), 0.0f);
        v_W_out_.resize(W_out_.size(), 0.0f);
    }

    void
    ClearGradients() {
        for (int32_t i = 0; i < num_layers_; ++i) {
            std::fill(d_fc_weights_[i].begin(), d_fc_weights_[i].end(), 0.0f);
            std::fill(d_fc_biases_[i].begin(), d_fc_biases_[i].end(), 0.0f);
            std::fill(d_ln_gammas_[i].begin(), d_ln_gammas_[i].end(), 0.0f);
            std::fill(d_ln_betas_[i].begin(), d_ln_betas_[i].end(), 0.0f);
        }
        std::fill(dW_out_.begin(), dW_out_.end(), 0.0f);
    }

    // ========== Layer Operations (BLAS optimized) ==========

    // Linear: Y = X @ W.T + b
    // X: [batch, in_dim], W: [out_dim, in_dim], Y: [batch, out_dim]
    void
    LinearForward(const float* X, const float* W, const float* b, int32_t batch, int32_t in_dim, int32_t out_dim,
                  float* Y) {
        // Y = X @ W.T using cblas_sgemm
        // C = alpha * A * B + beta * C
        // Here: Y = 1.0 * X * W.T + 0.0 * Y
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, out_dim, in_dim, 1.0f, X, in_dim, W, in_dim, 0.0f,
                    Y, out_dim);

        // Add bias: Y[i,:] += b for each row
        for (int32_t i = 0; i < batch; ++i) {
            cblas_saxpy(out_dim, 1.0f, b, 1, Y + i * out_dim, 1);
        }
    }

    // Linear without bias: Y = X @ W.T
    void
    LinearForwardNoBias(const float* X, const float* W, int32_t batch, int32_t in_dim, int32_t out_dim, float* Y) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, out_dim, in_dim, 1.0f, X, in_dim, W, in_dim, 0.0f,
                    Y, out_dim);
    }

    // Linear backward
    // dW: [out_dim, in_dim], dY: [batch, out_dim], X: [batch, in_dim], W: [out_dim, in_dim]
    void
    LinearBackward(const float* X, const float* dY, const float* W, int32_t batch, int32_t in_dim, int32_t out_dim,
                   float* dW, float* db, float* dX) {
        // dW += dY.T @ X
        // dW[out_dim, in_dim] += dY.T[out_dim, batch] @ X[batch, in_dim]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim, in_dim, batch, 1.0f, dY, out_dim, X, in_dim, 1.0f,
                    dW, in_dim);

        // db += sum(dY, axis=0) - sum each column of dY
        for (int32_t j = 0; j < out_dim; ++j) {
            float sum = 0.0f;
            for (int32_t i = 0; i < batch; ++i) {
                sum += dY[i * out_dim + j];
            }
            db[j] += sum;
        }

        // dX = dY @ W
        // dX[batch, in_dim] = dY[batch, out_dim] @ W[out_dim, in_dim]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, in_dim, out_dim, 1.0f, dY, out_dim, W, in_dim,
                    0.0f, dX, in_dim);
    }

    // Linear backward without bias
    void
    LinearBackwardNoBias(const float* X, const float* dY, int32_t batch, int32_t in_dim, int32_t out_dim, float* dW,
                         float* dX) {
        // dW += dY.T @ X
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim, in_dim, batch, 1.0f, dY, out_dim, X, in_dim, 1.0f,
                    dW, in_dim);

        // dX = dY @ W_out_
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch, in_dim, out_dim, 1.0f, dY, out_dim, W_out_.data(),
                    in_dim, 0.0f, dX, in_dim);
    }

    // LayerNorm forward: Y = (X - mean) / std * gamma + beta
    void
    LayerNormForward(const float* X, const float* gamma, const float* beta, int32_t batch, int32_t dim, float* Y) {
        const float eps = 1e-5f;

        for (int32_t i = 0; i < batch; ++i) {
            const float* x_row = X + i * dim;
            float* y_row = Y + i * dim;

            float mean = 0.0f;
            for (int32_t j = 0; j < dim; ++j) {
                mean += x_row[j];
            }
            mean /= dim;

            float var = 0.0f;
            for (int32_t j = 0; j < dim; ++j) {
                float diff = x_row[j] - mean;
                var += diff * diff;
            }
            var /= dim;

            float inv_std = 1.0f / std::sqrt(var + eps);

            for (int32_t j = 0; j < dim; ++j) {
                y_row[j] = (x_row[j] - mean) * inv_std * gamma[j] + beta[j];
            }
        }
    }

    // LayerNorm backward
    void
    LayerNormBackward(const float* X, const float* dY, const float* gamma, int32_t batch, int32_t dim, float* dX,
                      float* dgamma, float* dbeta) {
        const float eps = 1e-5f;
        std::vector<float> x_norm(dim);

        for (int32_t i = 0; i < batch; ++i) {
            const float* x_row = X + i * dim;
            const float* dy_row = dY + i * dim;
            float* dx_row = dX + i * dim;

            float mean = 0.0f;
            for (int32_t j = 0; j < dim; ++j) {
                mean += x_row[j];
            }
            mean /= dim;

            float var = 0.0f;
            for (int32_t j = 0; j < dim; ++j) {
                float diff = x_row[j] - mean;
                var += diff * diff;
            }
            var /= dim;

            float inv_std = 1.0f / std::sqrt(var + eps);

            for (int32_t j = 0; j < dim; ++j) {
                x_norm[j] = (x_row[j] - mean) * inv_std;
            }

            for (int32_t j = 0; j < dim; ++j) {
                dgamma[j] += dy_row[j] * x_norm[j];
                dbeta[j] += dy_row[j];
            }

            float sum_dy_gamma = 0.0f;
            float sum_dy_gamma_xnorm = 0.0f;
            for (int32_t j = 0; j < dim; ++j) {
                float dy_g = dy_row[j] * gamma[j];
                sum_dy_gamma += dy_g;
                sum_dy_gamma_xnorm += dy_g * x_norm[j];
            }

            float scale = inv_std / dim;
            for (int32_t j = 0; j < dim; ++j) {
                dx_row[j] = scale * (dim * dy_row[j] * gamma[j] - sum_dy_gamma - x_norm[j] * sum_dy_gamma_xnorm);
            }
        }
    }

    // GELU forward: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    void
    GELUForward(const float* X, int32_t n, float* Y) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;

        for (int32_t i = 0; i < n; ++i) {
            float x = X[i];
            float x3 = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x3);
            Y[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }

    // GELU backward
    void
    GELUBackward(const float* X, const float* dY, int32_t n, float* dX) {
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        const float coeff3 = 3.0f * coeff;

        for (int32_t i = 0; i < n; ++i) {
            float x = X[i];
            float x2 = x * x;
            float x3 = x2 * x;
            float inner = sqrt_2_over_pi * (x + coeff * x3);
            float tanh_inner = std::tanh(inner);
            float sech2 = 1.0f - tanh_inner * tanh_inner;
            float d_inner = sqrt_2_over_pi * (1.0f + coeff3 * x2);
            dX[i] = dY[i] * (0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner);
        }
    }

    // Adam update
    void
    AdamUpdate(std::vector<float>& param, std::vector<float>& grad, std::vector<float>& m, std::vector<float>& v,
               float lr, float beta1, float beta2, float eps, float bc1, float bc2) {
        const size_t n = param.size();
        const float one_minus_beta1 = 1.0f - beta1;
        const float one_minus_beta2 = 1.0f - beta2;
        const float lr_bc1 = lr / bc1;

        for (size_t i = 0; i < n; ++i) {
            m[i] = beta1 * m[i] + one_minus_beta1 * grad[i];
            v[i] = beta2 * v[i] + one_minus_beta2 * grad[i] * grad[i];
            float v_hat = v[i] / bc2;
            param[i] -= lr_bc1 * m[i] / (std::sqrt(v_hat) + eps);
        }
    }
};

}  // namespace knowhere

#endif /* SIMPLE_MLP_H */
