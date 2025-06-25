#include <torch/extension.h>

// Forward declaration of the CUDA kernel launcher
// Forward declaration of the templated CUDA kernel launcher implementation
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
);

// C++ wrapper function that will be called from Python
void t5_block_forward(
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
    // Manually dispatch to the correct templated C++ function based on the input tensor's dtype
    // to avoid compiling for unused types like 'double'.
    if (hidden_states.scalar_type() == at::kFloat) {
        t5_block_forward_cuda_launcher_impl<float>(
            hidden_states, position_bias, attention_mask,
            norm1_weight, q_weight, k_weight, v_weight, o_weight,
            norm2_weight, wi0_weight, wi1_weight, wo_weight,
            output, workspace
        );
    } else if (hidden_states.scalar_type() == at::kBFloat16) {
        t5_block_forward_cuda_launcher_impl<at::BFloat16>(
            hidden_states, position_bias, attention_mask,
            norm1_weight, q_weight, k_weight, v_weight, o_weight,
            norm2_weight, wi0_weight, wi1_weight, wo_weight,
            output, workspace
        );
    } else {
        AT_ERROR("t5_block_forward only supports Float and BFloat16, but got ", hidden_states.scalar_type());
    }
}

// Binding the C++ function to the Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("t5_block_forward", &t5_block_forward, "T5 Block Forward (CUDA)");
}