#include "TCNNBridge.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/config.h>

struct TCNNModel : ITCNNModel {
    tcnn::TrainableModel model;

    TCNNModel(uint32_t inputSize, uint32_t outputSize, const tcnn::json& CONFIG) {
        model = tcnn::create_from_config(inputSize, outputSize, CONFIG);
        model.network->set_jit_fusion(tcnn::supports_jit_fusion());
    }

    void inference(const tcnn::GPUMatrixDynamic<float>& input,
                   tcnn::GPUMatrixDynamic<float>& output) {
        model.network->inference(input, output);
    }

    float training_step(const tcnn::GPUMatrixDynamic<float>& input,
                        const tcnn::GPUMatrix<float>& target) {
        auto ctx = model.trainer->training_step(input, target);
        return model.trainer->loss(*ctx);
    }
};

std::unique_ptr<ITCNNModel> create_model(uint32_t input_dims, uint32_t output_dims, const std::string& config) {
    return std::make_unique<TCNNModel>(input_dims, output_dims, tcnn::json::parse(config));
}