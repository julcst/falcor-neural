/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "NRC.h"
#include "Utils/CudaUtils.h"
#include "../TraceQueries/Query.slang"
#include "NRC.slang"

const char kOutputsToTextureFile[] = "RenderPasses/NRC/OutputsToTexture.cs.slang";

const nlohmann::json CONFIG {
    {"encoding", {
        {"otype", "Composite"},
        {"nested", {
            // Position is encoded with Hashgrid -> 32 dim
            {
                {"n_dims_to_encode", 3},
                {"otype", "Grid"},
                {"type", "Hash"},
                {"n_levels", 16},
                {"n_features_per_level", 2},
                {"log2_hashmap_size", 19},
                {"base_resolution", 16},
                {"per_level_scale", 2.0}, // In [1.26, 2]
                {"interpolation", "Linear"}
            },
            // Roughness: OneBlob -> 4 dim
            {
                {"n_dims_to_encode", 1},
                {"otype", "OneBlob"},
                {"n_bins", 4}
            },
            // wo is encoded with SphericalHarmonics -> 3²=9 dim
            {
                {"n_dims_to_encode", 3},
                {"otype", "SphericalHarmonics"},
                {"degree", 3}
            },
            // wn is encoded with SphericalHarmonics -> 3²=9 dim
            {
                {"n_dims_to_encode", 3},
                {"otype", "SphericalHarmonics"},
                {"degree", 3}
            },
            // Rest is identity: diff + spec -> 6 dim
            {
                {"otype", "Identity"}
            }
        }} // -> 60 dim
    }},
    {"network", {
        {"otype", "FullyFusedMLP"},
        // "otype", "CutlassMLP",
        {"activation", "ReLU"},
        {"output_activation", "None"},
        {"n_neurons", 64},
        {"n_hidden_layers", 2}
    }},
    {"loss", {
        {"otype", "RelativeL2Luminance"}
    }},
    {"optimizer", {
        {"otype", "EMA"},
        {"decay", 0.95},
        {"nested", {
            {"otype", "Adam"},
            {"learning_rate", 1e-2},
            {"beta1", 0.9},
            {"beta2", 0.99},
            {"epsilon", 1e-15},
            {"l2_reg", 1e-6},
            {"absolute_decay", 0},
            {"relative_decay", 0},
            {"adabound", false}
        }}
    }}
};

constexpr std::string_view inferenceKernel = R"(
__device__ __forceinline__ constexpr float3 operator*(const float3& a, const float3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ __forceinline__ constexpr float3 operator+(const float3& a, const float3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ constexpr float4 operator*(const float a, const float4& b) {
    return {a * b.x, a * b.y, a * b.z, a * b.w};
}

__device__ __forceinline__ constexpr float4 operator+=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__device__ __forceinline__ constexpr float4 make_float4(const float3& a, float w) {
    return {a.x, a.y, a.z, w};
}

__global__ void inference_kernel(
    int2 dim,
    float* inferenceInput, 
    float3* inferenceThroughput,
    bool raw,
    float4* image,
    const network_precision_t* __restrict__ params
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool valid = (x < dim.x) && (y < dim.y);

    const int i = y * dim.x + x;
    const int idxIn = i * NRC_INPUT_SIZE;

    // Pack input into hvec
    tcnn::vec<NRC_INPUT_SIZE> nerf_in;
    #pragma unroll
    for (int j = 0; j < NRC_INPUT_SIZE; j++)
        nerf_in[j] = valid ? inferenceInput[idxIn + j] : 0.0f;

    // Call tiny-cuda-nn model. All 32 threads of the warp must be active here.
    tcnn::vec<NRC_OUTPUT_SIZE> nerf_out = model_fun(nerf_in, params);

    if (!valid) return; // All threads must be active until now

    auto inference = make_float3(nerf_out[0], nerf_out[1], nerf_out[2]);

    const auto throughput = inferenceThroughput[i];

    if (throughput.x <= 0.0f && throughput.y <= 0.0f && throughput.z <= 0.0f) return;

    if (raw) {
        image[i] = make_float4(inference, 1.0f);
    } else {
        const auto diffuse = make_float3(nerf_in[8], nerf_in[9], nerf_in[10]);
        const auto specular = make_float3(nerf_in[11], nerf_in[12], nerf_in[13]);
        image[i] = make_float4(inference * (diffuse + specular) * throughput, 1.0f);
    }
}
)";

namespace
{
// Inputs
const std::string kInferenceInput = "inferenceInput";
const std::string kInferenceQueries = "inferenceQueries";
const std::string kTrainInput = "trainInput";
const std::string kTrainTarget = "trainTarget";
// Internal
const std::string kInferenceOutputFloat = "inferenceOutputFloat";
//Output
const std::string kOutput = "output";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, NRC>();
}

NRC::NRC(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    model = create_model(NRC_INPUT_SIZE, NRC_OUTPUT_SIZE, CONFIG.dump());
}

Properties NRC::getProperties() const
{
    return {};
}

RenderPassReflection NRC::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInferenceInput, "Inference inputs")
        .rawBuffer(mInferenceSize * sizeof(NRCInput))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared);
    reflector.addInput(kInferenceQueries, "Inference queries")
        .rawBuffer(mInferenceSize * sizeof(Query))
        .bindFlags(ResourceBindFlags::UnorderedAccess);
    reflector.addInput(kTrainInput, "Training inputs")
        .rawBuffer(mTrainSize * sizeof(NRCInput))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared);
    reflector.addInput(kTrainTarget, "Training target radiance")
        .rawBuffer(mTrainSize * sizeof(NRCOutput))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared);

    reflector.addInternal(kInferenceOutputFloat, "Inference output as float matrix")
        .rawBuffer(mInferenceSize * NRC_OUTPUT_SIZE * sizeof(float))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared);
    
    reflector.addOutput(kOutput, "Inference output")
        .texture2D(0, 0)
        .format(ResourceFormat::RGBA32Float);
    
    return reflector;
}

void NRC::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    const auto inferenceSize = compileData.connectedResources.getField(kInferenceInput)->getWidth() / sizeof(NRCInput);
    FALCOR_ASSERT_EQ(inferenceSize % tcnn::BATCH_SIZE_GRANULARITY, 0);
    const auto trainSize = compileData.connectedResources.getField(kTrainInput)->getWidth() / sizeof(NRCInput);
    FALCOR_ASSERT_EQ(trainSize % tcnn::BATCH_SIZE_GRANULARITY, 0);
    if (mInferenceSize != inferenceSize || mTrainSize != trainSize) {
        mInferenceSize = inferenceSize;
        mTrainSize = trainSize;
        FALCOR_CHECK(false, "Recompute buffer sizes"); // Force retry of reflect
    }
}

void NRC::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Prepare passes
    if (!mpOutputsToTexturePass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kOutputsToTextureFile);
        desc.csEntry("main");
        mpOutputsToTexturePass = ComputePass::create(mpDevice, desc);
    }

    pRenderContext->submit(true); // Because we will use Cuda next we first need to explicitly wait for the current command queue to finish

    // for (auto i : {10u, 50u, 1000u}) {
    //     const auto s = renderData[kTrainInput]->asBuffer()->getElement<NRCInput>(i);
    //     logInfo("TrainInput {}: pos={},{},{} diff={},{},{}", i, s.position.x, s.position.y, s.position.z, s.diffuse.x, s.diffuse.y, s.diffuse.z);
    // }

    // for (auto i : {10u, 50u, 1000u}) {
    //     const auto s = renderData[kInferenceInput]->asBuffer()->getElement<NRCInput>(i);
    //     logInfo("InferenceInput {}: pos={},{},{} diff={},{},{}", i, s.position.x, s.position.y, s.position.z, s.diffuse.x, s.diffuse.y, s.diffuse.z);
    // }

    tcnn::GPUMatrixDynamic trainInput {(float*) renderData[kTrainInput]->asBuffer()->getCudaMemory()->getMappedData(), NRC_INPUT_SIZE, mTrainSize};
    tcnn::GPUMatrixDynamic trainTarget {(float*) renderData[kTrainTarget]->asBuffer()->getCudaMemory()->getMappedData(), NRC_OUTPUT_SIZE, mTrainSize};
    tcnn::GPUMatrixDynamic inferenceInput {(float*) renderData[kInferenceInput]->asBuffer()->getCudaMemory()->getMappedData(), NRC_INPUT_SIZE, mInferenceSize};
    tcnn::GPUMatrixDynamic inferenceOutput {(float*) renderData[kInferenceOutputFloat]->asBuffer()->getCudaMemory()->getMappedData(), NRC_OUTPUT_SIZE, mInferenceSize};

    {
        uint32_t batchSize = mTrainSize / mTrainSteps;
        for (uint32_t offset = 0; offset < mTrainSize; offset += batchSize) {
            // TODO: Limit training to the samples generated in this step to improve performance
            float loss = model->training_step(trainInput.slice_cols(offset, batchSize), trainTarget.slice_cols(offset, batchSize));
            logInfo("Training loss: {}", loss);
            //lossHistory.push_back(loss);
        }
    }

    {
        model->inference(inferenceInput, inferenceOutput);
        
        // Copy to texture
        auto var = mpOutputsToTexturePass->getRootVar();
        var["gInferenceQueries"] = renderData[kInferenceQueries]->asBuffer();
        var["gInferenceOutput"] = renderData[kInferenceOutputFloat]->asBuffer();
        var["gOutput"] = renderData[kOutput]->asTexture();
        
        Falcor::uint2 resolution = { renderData[kOutput]->asTexture()->getWidth(), renderData[kOutput]->asTexture()->getHeight() };
        var["CB"]["gResolution"] = resolution;
        
        mpOutputsToTexturePass->execute(pRenderContext, resolution.x, resolution.y, 1);
    }
}

void NRC::renderUI(Gui::Widgets& widget) {}
