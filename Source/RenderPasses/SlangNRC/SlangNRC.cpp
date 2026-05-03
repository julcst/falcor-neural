#include "SlangNRC.h"
#include "NRCCompat.slang"
#include "../TraceQueries/Query.slang"
#include "Config.slang"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{
const std::string kTrainShaderFile = "RenderPasses/SlangNRC/Training.cs.slang";
const std::string kOptimizeShaderFile = "RenderPasses/SlangNRC/Optimization.cs.slang";
const std::string kInferShaderFile = "RenderPasses/SlangNRC/Inference.cs.slang";

const std::string kTrainEntry = "trainMain";
const std::string kOptimizeEntry = "optimizeMain";
const std::string kResetEntry = "resetMain";
const std::string kInferEntry = "inferMain";

const std::string kInferenceInput = "inferenceInput";
const std::string kInferenceQueries = "inferenceQueries";
const std::string kTrainInput = "trainInput";
const std::string kTrainTarget = "trainTarget";
const std::string kOutput = "output";

const std::string kParams = "params";
const std::string kParamGrads = "paramGrads";
const std::string kMoments1 = "moments1";
const std::string kMoments2 = "moments2";
const std::string kEncodingParams = "encodingParams";
const std::string kEncodingParamGrads = "encodingParamGrads";
const std::string kEncodingMoments1 = "encodingMoments1";
const std::string kEncodingMoments2 = "encodingMoments2";

const std::string kTrainingSteps = "trainingSteps";
const std::string kLearningRate = "learningRate";
const std::string kUseFactorization = "useFactorization";
const std::string kOutputRaw = "outputRaw";

constexpr uint32_t kOptimizeDispatchThreads = (std::min(1u << 19u, MLPConfig::kEncodingParamElementCount) + 255u) / 256u * 256u;
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SlangNRC>();
}

SlangNRC::SlangNRC(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    setProperties(props);
}

void SlangNRC::setProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kTrainingSteps) mTrainSteps = value;
        else if (key == kLearningRate) mLearningRate = value;
        else if (key == kUseFactorization) mUseFactorization = value;
        else if (key == kOutputRaw) mOutputRaw = value;
        else logWarning("{} - Unrecognized property '{}'", getClassName(), key);
    }

    mTrainSteps = std::max(1u, mTrainSteps);
    mLearningRate = std::max(1e-6f, mLearningRate);
}

Properties SlangNRC::getProperties() const
{
    Properties props;
    props[kTrainingSteps] = mTrainSteps;
    props[kLearningRate] = mLearningRate;
    props[kUseFactorization] = mUseFactorization;
    props[kOutputRaw] = mOutputRaw;
    return props;
}

void SlangNRC::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Use Factorization", mUseFactorization);
    widget.checkbox("Raw Output", mOutputRaw);
    widget.var("Training Steps", mTrainSteps, 1u, 8u);
    widget.var("Learning Rate", mLearningRate, 0.0f, 1e-2f);
    if (widget.button("Reset")) mReset = true;
}

RenderPassReflection SlangNRC::reflect(const CompileData& compileData)
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

    // Inference output is written directly to the output texture by the merged infer pass.

    reflector.addInternal(kParams, "Persistent MLP parameters")
        .rawBuffer(MLPConfig::kParamBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addInternal(kParamGrads, "Persistent MLP parameter gradients")
        .rawBuffer(MLPConfig::kParamBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addInternal(kMoments1, "Persistent Adam moment1")
        .rawBuffer(MLPConfig::kMomentsBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addInternal(kMoments2, "Persistent Adam moment2")
        .rawBuffer(MLPConfig::kMomentsBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addInternal(kEncodingParams, "Persistent hash-grid encoding parameters")
        .rawBuffer(MLPConfig::kEncodingParamBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addInternal(kEncodingParamGrads, "Persistent hash-grid encoding parameter gradients")
        .rawBuffer(MLPConfig::kEncodingGradBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addInternal(kEncodingMoments1, "Persistent hash-grid Adam moment1")
        .rawBuffer(MLPConfig::kEncodingMomentsBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addInternal(kEncodingMoments2, "Persistent hash-grid Adam moment2")
        .rawBuffer(MLPConfig::kEncodingMomentsBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addOutput(kOutput, "Inference output")
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(0, 0);

    return reflector;
}

void SlangNRC::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    const uint32_t queryCount = compileData.connectedResources.getField(kInferenceQueries)->getWidth() / sizeof(Query);
    const uint32_t inferenceSize = compileData.connectedResources.getField(kInferenceInput)->getWidth() / sizeof(NRCInput);
    const uint32_t trainSize = compileData.connectedResources.getField(kTrainInput)->getWidth() / sizeof(NRCInput);

    FALCOR_ASSERT_EQ(inferenceSize % BATCH_SIZE_GRANULARITY, 0u);
    FALCOR_ASSERT_EQ(trainSize % BATCH_SIZE_GRANULARITY, 0u);

    if (mInferenceQueryCount != queryCount || mInferenceSize != inferenceSize || mTrainSize != trainSize)
    {
        mInferenceQueryCount = queryCount;
        mInferenceSize = inferenceSize;
        mTrainSize = trainSize;
        FALCOR_CHECK(false, "Recompute buffer sizes");
    }
}

void SlangNRC::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "SlangNRC");

    const auto pInferenceInput = renderData.getBuffer(kInferenceInput);
    const auto pInferenceQueries = renderData.getBuffer(kInferenceQueries);
    const auto pTrainInput = renderData.getBuffer(kTrainInput);
    const auto pTrainTarget = renderData.getBuffer(kTrainTarget);
    const auto pOutput = renderData.getTexture(kOutput);
    const auto pParams = renderData.getBuffer(kParams);
    const auto pParamGrads = renderData.getBuffer(kParamGrads);
    const auto pMoments1 = renderData.getBuffer(kMoments1);
    const auto pMoments2 = renderData.getBuffer(kMoments2);
    const auto pEncodingParams = renderData.getBuffer(kEncodingParams);
    const auto pEncodingParamGrads = renderData.getBuffer(kEncodingParamGrads);
    const auto pEncodingMoments1 = renderData.getBuffer(kEncodingMoments1);
    const auto pEncodingMoments2 = renderData.getBuffer(kEncodingMoments2);
    FALCOR_ASSERT(pInferenceInput && pInferenceQueries && pTrainInput && pTrainTarget && pOutput && pParams && pParamGrads && pMoments1 && pMoments2 && pEncodingParams && pEncodingParamGrads && pEncodingMoments1 && pEncodingMoments2);

    createPasses();

    if (mReset)
    {
        FALCOR_PROFILE(pRenderContext, "Reset");
        auto var = mpResetPass->getRootVar();
        var["gParams"] = pParams;
        var["gMoments1"] = pMoments1;
        var["gMoments2"] = pMoments2;
        var["gEncodingParams"] = pEncodingParams;
        var["gEncodingMoments1"] = pEncodingMoments1;
        var["gEncodingMoments2"] = pEncodingMoments2;
        var["CB"]["gDispatchThreadCount"] = kOptimizeDispatchThreads;

        mpResetPass->execute(pRenderContext, kOptimizeDispatchThreads, 1u, 1u);
        mReset = false;
    }

    // Factorization is handled inline inside the training shader when enabled.

    const uint32_t batchSize = std::max(1u, (mTrainSize + mTrainSteps - 1u) / mTrainSteps);
    for (uint32_t step = 0; step < mTrainSteps; ++step)
    {
        {
            FALCOR_PROFILE(pRenderContext, "Training");
            const uint32_t batchOffset = std::min(step * batchSize, mTrainSize);
            const uint32_t currentBatchSize = std::min(batchSize, mTrainSize - batchOffset);
            if (currentBatchSize == 0u) break;

            // Ensure training shader knows whether to apply factorization inline.
            mpTrainPass->addDefine("USE_FACTORISATION", mUseFactorization ? "1" : "0");
            auto var = mpTrainPass->getRootVar();
            var["gTrainInput"] = pTrainInput;
            var["gTrainTarget"] = pTrainTarget;
            var["gParams"] = pParams;
            var["gParamGrads"] = pParamGrads;
            var["gEncodingParams"] = pEncodingParams;
            var["gEncodingParamGrads"] = pEncodingParamGrads;

            var["CB"]["gBatchOffset"] = batchOffset;
            var["CB"]["gBatchSize"] = currentBatchSize;
            var["CB"]["gCurrentStep"] = mOptimizeStep;

            mpTrainPass->execute(pRenderContext, currentBatchSize, 1u, 1u);
        }

        {
            FALCOR_PROFILE(pRenderContext, "Optimization");
            auto optimizeVar = mpOptimizePass->getRootVar();
            optimizeVar["gParams"] = pParams;
            optimizeVar["gParamGrads"] = pParamGrads;
            optimizeVar["gMoments1"] = pMoments1;
            optimizeVar["gMoments2"] = pMoments2;
            optimizeVar["gEncodingParams"] = pEncodingParams;
            optimizeVar["gEncodingParamGrads"] = pEncodingParamGrads;
            optimizeVar["gEncodingMoments1"] = pEncodingMoments1;
            optimizeVar["gEncodingMoments2"] = pEncodingMoments2;
            optimizeVar["CB"]["gLearningRate"] = mLearningRate;
            optimizeVar["CB"]["gCurrentStep"] = float(mOptimizeStep);
            optimizeVar["CB"]["gDispatchThreadCount"] = kOptimizeDispatchThreads;

            mpOptimizePass->execute(pRenderContext, kOptimizeDispatchThreads, 1u, 1u);
            ++mOptimizeStep;
        }
    }

    {
        FALCOR_PROFILE(pRenderContext, "Inference");
        // Make the infer pass write the final texture directly. Pass factorization define as before.
        mpInferPass->addDefine("USE_FACTORISATION", (!mOutputRaw && mUseFactorization) ? "1" : "0");

        auto var = mpInferPass->getRootVar();
        var["gInferenceInput"] = pInferenceInput;
        var["gInferenceQueries"] = pInferenceQueries;
        var["gOutput"] = pOutput;
        var["gParams"] = pParams;
        var["gEncodingParams"] = pEncodingParams;
        var["CB"]["gCount"] = mInferenceSize;

        Falcor::uint2 resolution = { pOutput->getWidth(), pOutput->getHeight() };
        var["CB"]["gResolution"] = resolution;

        mpInferPass->execute(pRenderContext, mInferenceSize, 1u, 1u);
    }
}

void SlangNRC::createPasses()
{
    if (!mpResetPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kOptimizeShaderFile);
        desc.csEntry(kResetEntry);
        mpResetPass = ComputePass::create(mpDevice, desc);
    }


    if (!mpTrainPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kTrainShaderFile);
        desc.csEntry(kTrainEntry);
        mpTrainPass = ComputePass::create(mpDevice, desc);
    }

    if (!mpOptimizePass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kOptimizeShaderFile);
        desc.csEntry(kOptimizeEntry);
        mpOptimizePass = ComputePass::create(mpDevice, desc);
    }

    if (!mpInferPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kInferShaderFile);
        desc.csEntry(kInferEntry);
        mpInferPass = ComputePass::create(mpDevice, desc);
    }
}
