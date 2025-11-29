#pragma once
#include <memory>
#include <cstdint>

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/gpu_matrix.h>

// Pure virtual interface. 
class ITCNNModel {
public:
    // Virtual destructor is CRITICAL for unique_ptr to work correctly across boundaries
    virtual ~ITCNNModel() = default;
    virtual void inference(const tcnn::GPUMatrixDynamic<float>& input, tcnn::GPUMatrixDynamic<float>& output) = 0;
    virtual float training_step(const tcnn::GPUMatrixDynamic<float>& input, const tcnn::GPUMatrix<float>& target) = 0;
};

extern std::unique_ptr<ITCNNModel> create_model(uint32_t input_dims, uint32_t output_dims, const nlohmann::json& config);