#include "TCNNBridge.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/config.h>

#include <iostream>

using namespace tcnn;

struct TCNNModel : ITCNNModel {
    tcnn::TrainableModel model;
    cudaStream_t stream;

    TCNNModel(uint32_t inputSize, uint32_t outputSize, const tcnn::json& CONFIG, bool jitFusion) {
        model = tcnn::create_from_config(inputSize, outputSize, CONFIG);
        model.network->set_jit_fusion(jitFusion && tcnn::supports_jit_fusion());
        CUDA_CHECK_THROW(cudaStreamCreate(&stream));
        //CUDA_CHECK_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    ~TCNNModel() {
        CUDA_CHECK_THROW(cudaStreamDestroy(stream));
    }

    void inference(const tcnn::GPUMatrixDynamic<float>& input,
                   tcnn::GPUMatrixDynamic<float>& output) {
        model.network->inference(stream, input, output);
    }

    float training_step(const tcnn::GPUMatrixDynamic<float>& input,
                        const tcnn::GPUMatrix<float>& target) {
        auto ctx = model.trainer->training_step(stream, input, target);
        return model.trainer->loss(*ctx);
    }

    cudaStream_t getStream() {
        return stream;
    }
};

std::unique_ptr<ITCNNModel> create_model(uint32_t input_dims, uint32_t output_dims, const std::string& config, bool jitFusion) {
    return std::make_unique<TCNNModel>(input_dims, output_dims, tcnn::json::parse(config), jitFusion);
}