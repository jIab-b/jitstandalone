#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// T5 uses RMSNorm. This kernel computes it for a single row (token embedding).
template <typename T>
__global__ void rms_norm_kernel(
    T* out,
    const T* inp,
    const T* weight,
    int hidden_size,
    float epsilon
) {
    // Shared memory for the reduction
    extern __shared__ float s_mean[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int row_idx = blockIdx.x;

    // 1. Calculate sum of squares for the row
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += block_size) {
        float val = static_cast<float>(inp[row_idx * hidden_size + i]);
        sum_sq += val * val;
    }

    // 2. Reduce sum of squares in shared memory
    s_mean[tid] = sum_sq;
    __syncthreads();

    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_mean[tid] += s_mean[tid + s];
        }
        __syncthreads();
    }

    // 3. Finalize mean, compute rsqrt, and apply normalization
    if (tid == 0) {
        float mean = s_mean[0] / hidden_size;
        s_mean[0] = rsqrtf(mean + epsilon);
    }
    __syncthreads();

    float inv_std = s_mean[0];
    for (int i = tid; i < hidden_size; i += block_size) {
        float val = static_cast<float>(inp[row_idx * hidden_size + i]);
        float weight_val = static_cast<float>(weight[i]);
        out[row_idx * hidden_size + i] = static_cast<T>(val * inv_std * weight_val);
    }
}


// Simplified kernel for a linear layer (matrix multiplication)
template <typename T>
__global__ void linear_kernel(T* out, const T* inp, const T* weight, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += static_cast<float>(inp[row * N + i]) * static_cast<float>(weight[i * K + col]);
        }
        out[row * K + col] = static_cast<T>(sum);
    }
}

// Kernel to calculate attention scores: (Q @ K.T) / sqrt(d_k) + bias
template <typename T>
__global__ void attention_scores_kernel(T* out, const T* q, const T* k, const T* bias, int batch_size, int num_heads, int seq_len, int d_k) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x; // target seq_len
    int j = threadIdx.y; // source seq_len

    if (i < seq_len && j < seq_len) {
        float score = 0.0f;
        int q_offset = b * num_heads * seq_len * d_k + h * seq_len * d_k + i * d_k;
        int k_offset = b * num_heads * seq_len * d_k + h * seq_len * d_k + j * d_k;

        for (int d = 0; d < d_k; ++d) {
            score += static_cast<float>(q[q_offset + d]) * static_cast<float>(k[k_offset + d]);
        }
        score *= rsqrtf(static_cast<float>(d_k));
        if (bias != nullptr) {
             score += static_cast<float>(bias[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j]);
        }
        out[b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j] = static_cast<T>(score);
    }
}

// Numerically stable softmax for each row
template <typename T>
__global__ void softmax_kernel(T* data, int stride, int len) {
    int row_start_idx = blockIdx.x * stride;
    int tid = threadIdx.x;

    extern __shared__ float s_row[];
    float max_val = -FLT_MAX;
    for (int i = tid; i < len; i += blockDim.x) {
        max_val = max(max_val, static_cast<float>(data[row_start_idx + i]));
    }

    // Reduce max_val in shared memory
    // ... (standard block reduction) ...

    float sum = 0.0f;
    for (int i = tid; i < len; i += blockDim.x) {
        float val = expf(static_cast<float>(data[row_start_idx + i]) - max_val);
        sum += val;
        s_row[i] = val;
    }
     // Reduce sum in shared memory
    // ... (standard block reduction) ...

    for (int i = tid; i < len; i += blockDim.x) {
        data[row_start_idx + i] = static_cast<T>(s_row[i] / sum);
    }
}

// Kernel for the residual connection
template <typename T>
__global__ void residual_add_kernel(T* out, const T* inp1, const T* inp2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = inp1[idx] + inp2[idx];
    }
}

// The C++ launcher function, orchestrating the T5 block operations.
template <typename T>
void t5_block_forward_cuda_launcher_impl(
    const torch::Tensor& hidden_states,
    const torch::Tensor& position_bias,
    const torch::Tensor& attention_mask,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& v_weight,
    const torch::Tensor& o_weight,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& wi0_weight,
    const torch::Tensor& wi1_weight,
    const torch::Tensor& wo_weight,
    torch::Tensor& output,
    torch::Tensor& workspace
) {
    // --- Get Dimensions ---
    const auto batch_size = hidden_states.size(0);
    const auto seq_len = hidden_states.size(1);
    const auto hidden_size = hidden_states.size(2);
    const auto num_heads = q_weight.size(0) / hidden_size; // Infer from weights
    const auto d_k = hidden_size / num_heads;
    const auto ffn_dim = wi0_weight.size(1);

    const int threads_per_block = 256;
    const dim3 block_dim(threads_per_block, 1, 1);

    // --- Workspace Memory Layout ---
    T* current_ptr = workspace.data_ptr<T>();
    auto advance_ptr = [&](size_t elements) {
        T* ptr = current_ptr;
        current_ptr += elements;
        return ptr;
    };

    T* norm1_output = advance_ptr(batch_size * seq_len * hidden_size);
    T* q_proj = advance_ptr(batch_size * seq_len * hidden_size);
    T* k_proj = advance_ptr(batch_size * seq_len * hidden_size);
    T* v_proj = advance_ptr(batch_size * seq_len * hidden_size);
    T* attn_scores = advance_ptr(batch_size * num_heads * seq_len * seq_len);
    T* attn_output = advance_ptr(batch_size * seq_len * hidden_size);
    T* attn_residual = advance_ptr(batch_size * seq_len * hidden_size);
    T* norm2_output = advance_ptr(batch_size * seq_len * hidden_size);
    T* ffn_wi0 = advance_ptr(batch_size * seq_len * ffn_dim);
    T* ffn_wi1 = advance_ptr(batch_size * seq_len * ffn_dim);
    T* ffn_hidden = advance_ptr(batch_size * seq_len * ffn_dim);
    T* ffn_output = advance_ptr(batch_size * seq_len * hidden_size);


    // --- 1. First Layer Normalization (RMSNorm) ---
    const int blocks_per_grid_norm = batch_size * seq_len;
    const int shared_mem_size_norm = threads_per_block * sizeof(float);
    rms_norm_kernel<T><<<blocks_per_grid_norm, threads_per_block, shared_mem_size_norm>>>(
        norm1_output, hidden_states.data_ptr<T>(), norm1_weight.data_ptr<T>(), hidden_size, 1e-6f);

    // --- 2. Self-Attention ---
    // a. Project Q, K, V
    dim3 grid_dim_proj((hidden_size + 15) / 16, (batch_size * seq_len + 15) / 16, 1);
    dim3 block_dim_proj(16, 16, 1);
    linear_kernel<T><<<grid_dim_proj, block_dim_proj>>>(q_proj, norm1_output, q_weight.data_ptr<T>(), batch_size * seq_len, hidden_size, hidden_size);
    linear_kernel<T><<<grid_dim_proj, block_dim_proj>>>(k_proj, norm1_output, k_weight.data_ptr<T>(), batch_size * seq_len, hidden_size, hidden_size);
    linear_kernel<T><<<grid_dim_proj, block_dim_proj>>>(v_proj, norm1_output, v_weight.data_ptr<T>(), batch_size * seq_len, hidden_size, hidden_size);

    // b. Calculate attention scores
    dim3 grid_dim_scores(seq_len, num_heads, batch_size);
    dim3 block_dim_scores(32, 32, 1); // (target_len, source_len)
    attention_scores_kernel<T><<<grid_dim_scores, block_dim_scores>>>(
        attn_scores, q_proj, k_proj, position_bias.data_ptr<T>(), batch_size, num_heads, seq_len, d_k);

    // c. Apply softmax
    softmax_kernel<T><<<(batch_size * num_heads * seq_len), threads_per_block>>>(attn_scores, seq_len, seq_len);

    // d. Multiply scores by V
    // (Simplified matmul, in reality this is more complex)
    linear_kernel<T><<<grid_dim_proj, block_dim_proj>>>(attn_output, attn_scores, v_proj, batch_size * num_heads * seq_len, seq_len, d_k);

    // e. Output projection
    linear_kernel<T><<<grid_dim_proj, block_dim_proj>>>(
        ffn_output, attn_output, o_weight.data_ptr<T>(), batch_size * seq_len, hidden_size, hidden_size);

    // f. First residual connection
    residual_add_kernel<T><<<(hidden_states.numel() + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
        attn_residual, hidden_states.data_ptr<T>(), ffn_output, hidden_states.numel());


    // --- 3. Feed-Forward Network ---
    // a. Second Layer Normalization
    rms_norm_kernel<T><<<blocks_per_grid_norm, threads_per_block, shared_mem_size_norm>>>(
        norm2_output, attn_residual, norm2_weight.data_ptr<T>(), hidden_size, 1e-6f);

    // b. FFN projections (wi0 and wi1 for GLU)
    dim3 grid_dim_ffn((ffn_dim + 15) / 16, (batch_size * seq_len + 15) / 16, 1);
    linear_kernel<T><<<grid_dim_ffn, block_dim_proj>>>(ffn_wi0, norm2_output, wi0_weight.data_ptr<T>(), batch_size * seq_len, hidden_size, ffn_dim);
    linear_kernel<T><<<grid_dim_ffn, block_dim_proj>>>(ffn_wi1, norm2_output, wi1_weight.data_ptr<T>(), batch_size * seq_len, hidden_size, ffn_dim);

    // c. Apply GeGLU activation
    // (Element-wise multiply ffn_wi0 * ffn_wi1, simplified here)

    // d. Output projection (wo)
    linear_kernel<T><<<grid_dim_proj, block_dim_proj>>>(
        ffn_output, ffn_hidden, wo_weight.data_ptr<T>(), batch_size * seq_len, ffn_dim, hidden_size);

    // e. Final residual connection
    residual_add_kernel<T><<<(hidden_states.numel() + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
        output.data_ptr<T>(), attn_residual, ffn_output, hidden_states.numel());
}

// Forward declaration for the dispatcher in the binding file
void t5_block_forward_cuda_launcher(
    const torch::Tensor& hidden_states,
    const torch::Tensor& position_bias,
    const torch::Tensor& attention_mask,
    const torch::Tensor& norm1_weight,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    const torch::Tensor& v_weight,
    const torch::Tensor& o_weight,
    const torch::Tensor& norm2_weight,
    const torch::Tensor& wi0_weight,
    const torch::Tensor& wi1_weight,
    const torch::Tensor& wo_weight,
    torch::Tensor& output,
    torch::Tensor& workspace
);

// Explicitly instantiate the templates for the types we want to support.
template void t5_block_forward_cuda_launcher_impl<float>(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    torch::Tensor&, torch::Tensor&);

template void t5_block_forward_cuda_launcher_impl<at::BFloat16>(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    torch::Tensor&, torch::Tensor&);