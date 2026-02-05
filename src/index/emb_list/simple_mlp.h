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
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "knowhere/log.h"

// OpenBLAS thread control (defined in OpenBLAS library)
extern "C" {
void
openblas_set_num_threads(int num_threads);
int
openblas_get_num_threads(void);
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
        // Use pre-allocated buffers if available (training mode)
        const bool use_buffers = store_intermediates && (batch_size <= allocated_batch_size_);

        if (store_intermediates) {
            // Copy input to intermediates[0]
            if (use_buffers) {
                std::memcpy(intermediates_[0].data(), input, batch_size * input_dim_ * sizeof(float));
            } else {
                intermediates_.clear();
                pre_ln_values_.clear();
                post_ln_values_.clear();
                intermediates_.emplace_back(input, input + batch_size * input_dim_);
            }
        }

        // Current activation pointer
        const float* current = input;
        int32_t current_dim = input_dim_;

        // Forward through feature_extractor layers
        for (int32_t layer = 0; layer < num_layers_; ++layer) {
            int32_t out_dim = layer_dims_[layer];

            float* linear_out;
            float* ln_out;
            float* act_out;

            if (use_buffers) {
                linear_out = buf_linear_out_[layer].data();
                ln_out = buf_ln_out_[layer].data();
                act_out = buf_act_out_[layer].data();
            } else {
                // Fallback to dynamic allocation for inference
                buf_linear_out_.resize(std::max((int32_t)buf_linear_out_.size(), layer + 1));
                buf_ln_out_.resize(std::max((int32_t)buf_ln_out_.size(), layer + 1));
                buf_act_out_.resize(std::max((int32_t)buf_act_out_.size(), layer + 1));
                buf_linear_out_[layer].resize(batch_size * out_dim);
                buf_ln_out_[layer].resize(batch_size * out_dim);
                buf_act_out_[layer].resize(batch_size * out_dim);
                linear_out = buf_linear_out_[layer].data();
                ln_out = buf_ln_out_[layer].data();
                act_out = buf_act_out_[layer].data();
            }

            // Linear: out = input @ W.T + b
            LinearForward(current, fc_weights_[layer].data(), fc_biases_[layer].data(), batch_size, current_dim,
                          out_dim, linear_out);

            if (store_intermediates) {
                if (use_buffers) {
                    std::memcpy(pre_ln_values_[layer].data(), linear_out, batch_size * out_dim * sizeof(float));
                } else {
                    pre_ln_values_.emplace_back(linear_out, linear_out + batch_size * out_dim);
                }
            }

            // LayerNorm
            LayerNormForward(linear_out, ln_gammas_[layer].data(), ln_betas_[layer].data(), batch_size, out_dim,
                             ln_out);

            if (store_intermediates) {
                if (use_buffers) {
                    std::memcpy(post_ln_values_[layer].data(), ln_out, batch_size * out_dim * sizeof(float));
                } else {
                    post_ln_values_.emplace_back(ln_out, ln_out + batch_size * out_dim);
                }
            }

            // GELU activation
            GELUForward(ln_out, batch_size * out_dim, act_out);

            if (store_intermediates) {
                if (use_buffers) {
                    std::memcpy(intermediates_[layer + 1].data(), act_out, batch_size * out_dim * sizeof(float));
                } else {
                    intermediates_.emplace_back(act_out, act_out + batch_size * out_dim);
                }
            }

            current = act_out;
            current_dim = out_dim;
        }

        // Store final hidden for feature extraction
        if (use_buffers) {
            std::memcpy(final_hidden_.data(), current, batch_size * final_hidden_dim_ * sizeof(float));
        } else {
            final_hidden_.assign(current, current + batch_size * final_hidden_dim_);
        }

        // Output layer: out = hidden @ W_out.T (no bias)
        LinearForwardNoBias(current, W_out_.data(), batch_size, final_hidden_dim_, output_dim_, output);
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

        // Use pre-allocated buffers if available
        const bool use_buffers = (batch_size <= allocated_batch_size_);

        float total_loss = 0.0f;
        float scale = 1.0f / batch_size;

        // Get d_output buffer
        float* d_output;
        if (use_buffers) {
            d_output = buf_d_output_.data();
        } else {
            buf_d_output_.resize(batch_size * output_dim_);
            d_output = buf_d_output_.data();
        }

        // Compute output gradient: d_output = 2 * (output - target) / output_dim
        for (int32_t b = 0; b < batch_size; ++b) {
            for (int32_t j = 0; j < output_dim_; ++j) {
                float diff = output[b * output_dim_ + j] - target[b * output_dim_ + j];
                total_loss += diff * diff;
                d_output[b * output_dim_ + j] = 2.0f * diff * scale / output_dim_;
            }
        }

        // Get d_hidden buffer
        float* d_hidden;
        if (use_buffers) {
            d_hidden = buf_d_hidden_.data();
            std::fill(d_hidden, d_hidden + batch_size * final_hidden_dim_, 0.0f);
        } else {
            buf_d_hidden_.resize(batch_size * final_hidden_dim_);
            std::fill(buf_d_hidden_.begin(), buf_d_hidden_.end(), 0.0f);
            d_hidden = buf_d_hidden_.data();
        }

        // Backward through output layer (no bias)
        LinearBackwardNoBias(final_hidden_.data(), d_output, batch_size, final_hidden_dim_, output_dim_, dW_out_.data(),
                             d_hidden);

        // Pointer to current gradient (starts as d_hidden)
        float* d_current = d_hidden;

        // Backward through feature_extractor layers (reverse order)
        for (int32_t layer = num_layers_ - 1; layer >= 0; --layer) {
            int32_t out_dim = layer_dims_[layer];
            int32_t in_dim = (layer == 0) ? input_dim_ : layer_dims_[layer - 1];

            // Get stored intermediate values
            const float* layer_input = intermediates_[layer].data();
            const float* pre_ln = pre_ln_values_[layer].data();    // Linear output (before LayerNorm)
            const float* post_ln = post_ln_values_[layer].data();  // LayerNorm output (GELU input)

            // Get buffers for this layer
            float* d_act;
            float* d_ln;
            float* d_input_layer;

            if (use_buffers) {
                d_act = buf_d_act_[layer].data();
                d_ln = buf_d_ln_[layer].data();
                d_input_layer = buf_d_input_[layer].data();
                std::fill(d_input_layer, d_input_layer + batch_size * in_dim, 0.0f);
            } else {
                buf_d_act_.resize(std::max((int32_t)buf_d_act_.size(), layer + 1));
                buf_d_ln_.resize(std::max((int32_t)buf_d_ln_.size(), layer + 1));
                buf_d_input_.resize(std::max((int32_t)buf_d_input_.size(), layer + 1));
                buf_d_act_[layer].resize(batch_size * out_dim);
                buf_d_ln_[layer].resize(batch_size * out_dim);
                buf_d_input_[layer].resize(batch_size * in_dim, 0.0f);
                d_act = buf_d_act_[layer].data();
                d_ln = buf_d_ln_[layer].data();
                d_input_layer = buf_d_input_[layer].data();
            }

            // Backward GELU - use post_ln (LayerNorm output) as GELU's input
            GELUBackward(post_ln, d_current, batch_size * out_dim, d_act);

            // Backward LayerNorm
            LayerNormBackward(pre_ln, d_act, ln_gammas_[layer].data(), batch_size, out_dim, d_ln,
                              d_ln_gammas_[layer].data(), d_ln_betas_[layer].data());

            // Backward Linear
            LinearBackward(layer_input, d_ln, fc_weights_[layer].data(), batch_size, in_dim, out_dim,
                           d_fc_weights_[layer].data(), d_fc_biases_[layer].data(), d_input_layer);

            d_current = d_input_layer;
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
        // 8 threads is the sweet spot - more threads cause synchronization overhead
        int hw_threads = std::thread::hardware_concurrency();
        if (hw_threads < 1) {
            hw_threads = 4;
        }
        int num_threads = std::min(hw_threads, 8);
        openblas_set_num_threads(num_threads);

        // Pre-allocate all training buffers to avoid malloc in hot loop
        AllocateTrainingBuffers(batch_size);

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
        const int32_t patience = 5;     // Stop if no improvement for 5 epochs
        const float min_delta = 1e-4f;  // Minimum improvement to reset patience

        auto training_start = std::chrono::high_resolution_clock::now();

        for (int32_t epoch = 0; epoch < epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
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

            auto epoch_end = std::chrono::high_resolution_clock::now();
            double epoch_ms = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();

            // Log training progress
            bool should_log =
                (epoch == 0) || (epoch == epochs - 1) || (log_interval > 0 && (epoch + 1) % log_interval == 0);
            if (should_log) {
                LOG_KNOWHERE_INFO_ << "[LEMUR MLP] Epoch " << (epoch + 1) << "/" << epochs << ", Loss: " << final_loss
                                   << ", Time: " << epoch_ms << " ms";
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

    // ========== Pre-allocated buffers for training (avoid malloc in hot loop) ==========
    int32_t allocated_batch_size_ = 0;  // Current allocated batch size

    // Forward pass buffers (per layer)
    std::vector<std::vector<float>> buf_linear_out_;  // [num_layers][batch * layer_dim]
    std::vector<std::vector<float>> buf_ln_out_;      // [num_layers][batch * layer_dim]
    std::vector<std::vector<float>> buf_act_out_;     // [num_layers][batch * layer_dim]
    std::vector<float> buf_current_;                  // [batch * max_dim]

    // Backward pass buffers
    std::vector<float> buf_d_output_;              // [batch * output_dim]
    std::vector<float> buf_d_hidden_;              // [batch * final_hidden_dim]
    std::vector<std::vector<float>> buf_d_act_;    // [num_layers][batch * layer_dim]
    std::vector<std::vector<float>> buf_d_ln_;     // [num_layers][batch * layer_dim]
    std::vector<std::vector<float>> buf_d_input_;  // [num_layers][batch * in_dim]

    // LayerNorm backward buffer
    std::vector<float> buf_x_norm_;  // [max_dim]

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

    /**
     * @brief Pre-allocate buffers for training to avoid malloc in hot loop.
     */
    void
    AllocateTrainingBuffers(int32_t batch_size) {
        if (batch_size <= allocated_batch_size_) {
            return;  // Already allocated enough
        }

        allocated_batch_size_ = batch_size;

        // Find max dimension across all layers
        int32_t max_dim = input_dim_;
        for (int32_t i = 0; i < num_layers_; ++i) {
            max_dim = std::max(max_dim, layer_dims_[i]);
        }

        // Forward pass buffers
        buf_linear_out_.resize(num_layers_);
        buf_ln_out_.resize(num_layers_);
        buf_act_out_.resize(num_layers_);
        for (int32_t i = 0; i < num_layers_; ++i) {
            buf_linear_out_[i].resize(batch_size * layer_dims_[i]);
            buf_ln_out_[i].resize(batch_size * layer_dims_[i]);
            buf_act_out_[i].resize(batch_size * layer_dims_[i]);
        }
        buf_current_.resize(batch_size * max_dim);

        // Backward pass buffers
        buf_d_output_.resize(batch_size * output_dim_);
        buf_d_hidden_.resize(batch_size * final_hidden_dim_);
        buf_d_act_.resize(num_layers_);
        buf_d_ln_.resize(num_layers_);
        buf_d_input_.resize(num_layers_);
        for (int32_t i = 0; i < num_layers_; ++i) {
            int32_t in_dim = (i == 0) ? input_dim_ : layer_dims_[i - 1];
            buf_d_act_[i].resize(batch_size * layer_dims_[i]);
            buf_d_ln_[i].resize(batch_size * layer_dims_[i]);
            buf_d_input_[i].resize(batch_size * in_dim);
        }

        // LayerNorm buffer
        buf_x_norm_.resize(max_dim);

        // Pre-allocate intermediate storage for backward pass
        intermediates_.resize(num_layers_ + 1);
        pre_ln_values_.resize(num_layers_);
        post_ln_values_.resize(num_layers_);
        for (int32_t i = 0; i < num_layers_; ++i) {
            int32_t in_dim = (i == 0) ? input_dim_ : layer_dims_[i - 1];
            intermediates_[i].resize(batch_size * in_dim);
            pre_ln_values_[i].resize(batch_size * layer_dims_[i]);
            post_ln_values_[i].resize(batch_size * layer_dims_[i]);
        }
        intermediates_[num_layers_].resize(batch_size * final_hidden_dim_);
        final_hidden_.resize(batch_size * final_hidden_dim_);
    }

    // ========== Layer Operations (BLAS optimized) ==========

    // Linear: Y = X @ W.T + b
    // X: [batch, in_dim], W: [out_dim, in_dim], Y: [batch, out_dim]
    void
    LinearForward(const float* X, const float* W, const float* b, int32_t batch, int32_t in_dim, int32_t out_dim,
                  float* Y) {
        // Y = X @ W.T using cblas_sgemm
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, batch, out_dim, in_dim, 1.0f, X, in_dim, W, in_dim, 0.0f,
                    Y, out_dim);

        // Add bias: Y[i,:] += b - row-major traversal for cache efficiency
        // Instead of batch calls to cblas_saxpy, use single loop with better cache locality
        for (int32_t i = 0; i < batch; ++i) {
            float* y_row = Y + i * out_dim;
            for (int32_t j = 0; j < out_dim; ++j) {
                y_row[j] += b[j];
            }
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
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, out_dim, in_dim, batch, 1.0f, dY, out_dim, X, in_dim, 1.0f,
                    dW, in_dim);

        // db += sum(dY, axis=0) - row-major traversal for cache efficiency
        // Traverse row-by-row (contiguous memory access) instead of column-by-column
        for (int32_t i = 0; i < batch; ++i) {
            const float* dy_row = dY + i * out_dim;
            for (int32_t j = 0; j < out_dim; ++j) {
                db[j] += dy_row[j];
            }
        }

        // dX = dY @ W
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

        // Use pre-allocated buffer if available, otherwise use member buffer
        float* x_norm;
        if ((int32_t)buf_x_norm_.size() >= dim) {
            x_norm = buf_x_norm_.data();
        } else {
            buf_x_norm_.resize(dim);
            x_norm = buf_x_norm_.data();
        }

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

    // Fast tanh approximation using Pade approximation
    // Accurate to ~1e-4 in range [-3, 3], saturates outside
    static inline float
    fast_tanh(float x) {
        if (x < -3.0f)
            return -1.0f;
        if (x > 3.0f)
            return 1.0f;
        float x2 = x * x;
        return x * (27.0f + x2) / (27.0f + 9.0f * x2);
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
            Y[i] = 0.5f * x * (1.0f + fast_tanh(inner));
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
            float tanh_inner = fast_tanh(inner);
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
