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

namespace
{
const char kOutputsToTextureFile[] = "RenderPasses/NRC/OutputsToTexture.cs.slang";
const char kFactorizeOutputFile[] = "RenderPasses/NRC/FactorizeOutput.cs.slang";

// Inputs
const std::string kInferenceInput = "inferenceInput";
const std::string kInferenceQueries = "inferenceQueries";
const std::string kTrainInput = "trainInput";
const std::string kTrainTarget = "trainTarget";

// Internal
const std::string kInferenceOutputFloat = "inferenceOutputFloat";

// Output
const std::string kOutput = "output";

// Config
const std::string kUseFactorization = "useFactorization";
const std::string kOutputRaw = "outputRaw";
const std::string kTrainingSteps = "trainingSteps";
const std::string kJITFusion = "jitFusion";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, NRC>();
}

NRC::NRC(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    setProperties(props);
    model = create_model(NRC_INPUT_SIZE, NRC_OUTPUT_SIZE, CONFIG.dump(), mJitFusion);
}

void NRC::setProperties(const Properties& props)
{
    for (const auto& [key, value] : props) {
        if (key == kUseFactorization) mUseFactorization = value;
        else if (key == kOutputRaw) mOutputRaw = value;
        else if (key == kTrainingSteps) mTrainSteps = value;
        else if (key == kJITFusion) mJitFusion = value;
        else logWarning("{}: Unknown property '{}'", getClassName(), key);
    }
}

Properties NRC::getProperties() const
{
    Properties props;
    props[kUseFactorization] = mUseFactorization;
    props[kOutputRaw] = mOutputRaw;
    props[kTrainingSteps] = mTrainSteps;
    props[kJITFusion] = mJitFusion;
    return props;
}

void NRC::renderUI(Gui::Widgets& widget) {
    widget.checkbox("Use Factorization", mUseFactorization);
    widget.checkbox("Raw Output", mOutputRaw);
    widget.var("Training Steps", mTrainSteps, 1u, 8u);
    if (widget.checkbox("JIT Fusion", mJitFusion)) {
        model = create_model(NRC_INPUT_SIZE, NRC_OUTPUT_SIZE, CONFIG.dump(), mJitFusion);
    }
}

RenderPassReflection NRC::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInferenceInput, "Inference inputs")
        .rawBuffer(mInferenceSize * sizeof(NRCInput))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared);
    reflector.addInput(kInferenceQueries, "Inference queries")
        .rawBuffer(mInferenceQueryCount * sizeof(Query))
        .bindFlags(ResourceBindFlags::UnorderedAccess);
    reflector.addInput(kTrainInput, "Training inputs")
        .rawBuffer(mTrainSize * sizeof(NRCInput))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared);
    reflector.addInput(kTrainTarget, "Training target radiance")
        .rawBuffer(mTrainSize * sizeof(NRCOutput))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared);

    reflector.addInternal(kInferenceOutputFloat, "Inference output as float matrix")
        .rawBuffer(mInferenceSize * sizeof(NRCOutput))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared);
    
    reflector.addOutput(kOutput, "Inference output")
        .texture2D(0, 0)
        .format(ResourceFormat::RGBA32Float);
    
    return reflector;
}

void NRC::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    const uint32_t queryCount = compileData.connectedResources.getField(kInferenceQueries)->getWidth() / sizeof(Query);
    const uint32_t inferenceSize = compileData.connectedResources.getField(kInferenceInput)->getWidth() / sizeof(NRCInput);
    FALCOR_ASSERT_EQ(inferenceSize % NRC_BATCH_SIZE_GRANULARITY, 0);
    const uint32_t trainSize = compileData.connectedResources.getField(kTrainInput)->getWidth() / sizeof(NRCInput);
    FALCOR_ASSERT_EQ(trainSize % NRC_BATCH_SIZE_GRANULARITY, 0);
    if (mInferenceSize != inferenceSize || mTrainSize != trainSize || mInferenceQueryCount != queryCount) {
        mInferenceQueryCount = queryCount;
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

    if (!mpFactorizeOutputPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kFactorizeOutputFile);
        desc.csEntry("main");
        mpFactorizeOutputPass = ComputePass::create(mpDevice, desc);
    }

    if (mUseFactorization) {
        FALCOR_PROFILE(pRenderContext, "FactorizeTrainingData");
        auto var = mpFactorizeOutputPass->getRootVar();
        var["gInput"] = renderData.getBuffer(kTrainInput);
        var["gOutput"] = renderData.getBuffer(kTrainTarget);
        
        var["CB"]["gCount"] = mTrainSize;
        
        mpFactorizeOutputPass->execute(pRenderContext, mTrainSize, 1);
    }

    // for (auto i : {10u, 50u, 1000u}) {
    //     const auto s = renderData[kTrainInput]->asBuffer()->getElement<NRCInput>(i);
    //     logInfo("TrainInput {}: pos={},{},{} diff={},{},{}", i, s.position.x, s.position.y, s.position.z, s.diffuse.x, s.diffuse.y, s.diffuse.z);
    // }

    // for (auto i : {10u, 50u, 1000u}) {
    //     const auto s = renderData[kInferenceInput]->asBuffer()->getElement<NRCInput>(i);
    //     logInfo("InferenceInput {}: pos={},{},{} diff={},{},{}", i, s.position.x, s.position.y, s.position.z, s.diffuse.x, s.diffuse.y, s.diffuse.z);
    // }

    pRenderContext->waitForFalcor(model->getStream());

    {
        FALCOR_PROFILE(pRenderContext, "Training");
        tcnn::GPUMatrixDynamic trainInput {(float*) renderData.getBuffer(kTrainInput)->getCudaMemory()->getMappedData(), NRC_INPUT_SIZE, mTrainSize};
        tcnn::GPUMatrixDynamic trainTarget {(float*) renderData.getBuffer(kTrainTarget)->getCudaMemory()->getMappedData(), NRC_OUTPUT_SIZE, mTrainSize};
        uint32_t batchSize = mTrainSize / mTrainSteps;
        for (uint32_t offset = 0; offset < mTrainSize; offset += batchSize) {
            // TODO: Limit training to the samples generated in this step to improve performance
            float loss = model->training_step(trainInput.slice_cols(offset, batchSize), trainTarget.slice_cols(offset, batchSize));
            logInfo("Training loss: {}", loss);
            //lossHistory.push_back(loss);
        }
        pRenderContext->waitForCuda(model->getStream()); // NOTE: CPU wait on Vulkan
    }

    {
        FALCOR_PROFILE(pRenderContext, "Inference");
        tcnn::GPUMatrixDynamic inferenceInput {(float*) renderData[kInferenceInput]->asBuffer()->getCudaMemory()->getMappedData(), NRC_INPUT_SIZE, mInferenceSize};
        tcnn::GPUMatrixDynamic inferenceOutput {(float*) renderData[kInferenceOutputFloat]->asBuffer()->getCudaMemory()->getMappedData(), NRC_OUTPUT_SIZE, mInferenceSize};
        model->inference(inferenceInput, inferenceOutput);
        pRenderContext->waitForCuda(model->getStream()); // NOTE: CPU wait on Vulkan
    }

    {
        FALCOR_PROFILE(pRenderContext, "OutputsToTexture");
        // Copy to texture
        mpOutputsToTexturePass->addDefine("USE_FACTORISATION", (!mOutputRaw && mUseFactorization) ? "1" : "0");
        auto var = mpOutputsToTexturePass->getRootVar();
        var["gInferenceQueries"] = renderData.getBuffer(kInferenceQueries);
        var["gInferenceOutput"] = renderData.getBuffer(kInferenceOutputFloat);
        var["gInferenceInput"] = renderData.getBuffer(kInferenceInput);
        var["gOutput"] = renderData.getTexture(kOutput);
        
        Falcor::uint2 resolution = { renderData.getTexture(kOutput)->getWidth(), renderData.getTexture(kOutput)->getHeight() };
        var["CB"]["gResolution"] = resolution;
        
        mpOutputsToTexturePass->execute(pRenderContext, resolution.x, resolution.y, 1);
    }

    // NOTE: Currently Cuda GPU time spills to other passes making profiling difficult
}
