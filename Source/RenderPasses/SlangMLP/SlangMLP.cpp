#include "SlangMLP.h"
#include "SlangMLPConfig.slang"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{
const char kShaderFile[] = "RenderPasses/SlangMLP/SlangMLP.cs.slang";
const char kTrainEntry[] = "trainMain";
const char kOptimizeEntry[] = "optimizeMain";
const char kInferEntry[] = "inferMain";

const std::string kInput = "src";
const std::string kOutput = "dst";
const std::string kParams = "params";
const std::string kParamGrads = "paramGrads";
const std::string kMoments1 = "moments1";
const std::string kMoments2 = "moments2";

const char kTrainSteps[] = "trainSteps";
const char kBatchSize[] = "batchSize";
const char kLearningRate[] = "learningRate";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SlangMLP>();
}

SlangMLP::SlangMLP(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    setProperties(props);
}

void SlangMLP::setProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kTrainSteps)
            mTrainSteps = value;
        else if (key == kBatchSize)
            mBatchSize = value;
        else if (key == kLearningRate)
            mLearningRate = value;
        else
            logWarning("{} - Unrecognized property '{}'", getClassName(), key);
    }

    mTrainSteps = std::max(1u, mTrainSteps);
    mBatchSize = std::max(1u, mBatchSize);
    mLearningRate = std::max(1e-6f, mLearningRate);
}

void SlangMLP::renderUI(Gui::Widgets& widget)
{
    widget.var("Train steps", mTrainSteps, 1u, 4096u);
    widget.var("Batch size", mBatchSize, 1u, 1u << 20u);
    widget.var("Learning rate", mLearningRate, 1e-5f, 1.0f, 1e-5f, true);
    if (widget.button("Reset"))
        mReset = true;
}

Properties SlangMLP::getProperties() const
{
    Properties props;
    props[kTrainSteps] = mTrainSteps;
    props[kBatchSize] = mBatchSize;
    props[kLearningRate] = mLearningRate;
    return props;
}

RenderPassReflection SlangMLP::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInput, "Input image to approximate")
        .bindFlags(ResourceBindFlags::ShaderResource)
        .texture2D(0, 0);
    reflector.addInternal(kParams, "Persistent MLP parameters")
        .rawBuffer(SlangMLPConfig::kParamBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addInternal(kParamGrads, "Persistent MLP parameter gradients")
        .rawBuffer(SlangMLPConfig::kParamBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addInternal(kMoments1, "Persistent Adam moment1")
        .rawBuffer(SlangMLPConfig::kMomentsBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addInternal(kMoments2, "Persistent Adam moment2")
        .rawBuffer(SlangMLPConfig::kMomentsBufferSize)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addOutput(kOutput, "MLP approximation")
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(0, 0);

    return reflector;
}

void SlangMLP::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "SlangMLP");

    const auto pInput = renderData.getTexture(kInput);
    const auto pOutput = renderData.getTexture(kOutput);
    const auto pParams = renderData.getBuffer(kParams);
    const auto pParamGrads = renderData.getBuffer(kParamGrads);
    const auto pMoments1 = renderData.getBuffer(kMoments1);
    const auto pMoments2 = renderData.getBuffer(kMoments2);
    FALCOR_ASSERT(pInput && pOutput && pParams && pParamGrads && pMoments1 && pMoments2);

    createPasses();

    const uint32_t trainIterations = mReset ? 1u : mTrainSteps;
    for (uint32_t i = 0; i < trainIterations; ++i)
    {
        auto var = mpTrainPass->getRootVar();
        var["gInput"] = pInput;
        var["gOutput"] = pOutput;
        var["gParams"] = pParams;
        var["gParamsRW"] = pParams;
        var["gParamGrads"] = pParamGrads;
        var["gMoments1"] = pMoments1;
        var["gMoments2"] = pMoments2;

        var["CB"]["gFrameDim"] = uint2(pInput->getWidth(), pInput->getHeight());
        var["CB"]["gFrameIndex"] = mFrameIndex;
        var["CB"]["gTrainSteps"] = mTrainSteps;
        var["CB"]["gBatchSize"] = mBatchSize;
        var["CB"]["gLearningRate"] = mLearningRate;
        var["CB"]["gReset"] = mReset ? 1u : 0u;
        var["CB"]["gCurrentStep"] = mOptimizeStep;

        mpTrainPass->execute(pRenderContext, mBatchSize, 1u, 1u);

        auto optimizeVar = mpOptimizePass->getRootVar();
        optimizeVar["gInput"] = pInput;
        optimizeVar["gOutput"] = pOutput;
        optimizeVar["gParams"] = pParams;
        optimizeVar["gParamsRW"] = pParams;
        optimizeVar["gParamGrads"] = pParamGrads;
        optimizeVar["gMoments1"] = pMoments1;
        optimizeVar["gMoments2"] = pMoments2;
        optimizeVar["CB"]["gFrameDim"] = uint2(pInput->getWidth(), pInput->getHeight());
        optimizeVar["CB"]["gFrameIndex"] = mFrameIndex;
        optimizeVar["CB"]["gTrainSteps"] = mTrainSteps;
        optimizeVar["CB"]["gLearningRate"] = mLearningRate;
        optimizeVar["CB"]["gReset"] = mReset ? 1u : 0u;
        optimizeVar["CB"]["gCurrentStep"] = mOptimizeStep;

        mpOptimizePass->execute(pRenderContext, SlangMLPConfig::kParamElementCount, 1u, 1u);
        if (!mReset)
            ++mOptimizeStep;
    }

    {
        auto var = mpInferPass->getRootVar();
        var["gInput"] = pInput;
        var["gParams"] = pParams;
        var["gParamsRW"] = pParams;
        var["gParamGrads"] = pParamGrads;
        var["gMoments1"] = pMoments1;
        var["gMoments2"] = pMoments2;
        var["gOutput"] = pOutput;
        var["CB"]["gFrameDim"] = uint2(pOutput->getWidth(), pOutput->getHeight());
        var["CB"]["gFrameIndex"] = mFrameIndex;
        var["CB"]["gTrainSteps"] = mTrainSteps;
        var["CB"]["gLearningRate"] = mLearningRate;
        var["CB"]["gReset"] = 0u;
        var["CB"]["gCurrentStep"] = mOptimizeStep;

        mpInferPass->execute(pRenderContext, pOutput->getWidth(), pOutput->getHeight(), 1u);
    }

    if (mReset)
    {
        mReset = false;
        mOptimizeStep = 1;
    }
    ++mFrameIndex;
}

void SlangMLP::createPasses()
{
    if (!mpTrainPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderFile);
        desc.csEntry(kTrainEntry);
        mpTrainPass = ComputePass::create(mpDevice, desc);
    }

    if (!mpOptimizePass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderFile);
        desc.csEntry(kOptimizeEntry);
        mpOptimizePass = ComputePass::create(mpDevice, desc);
    }

    if (!mpInferPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderFile);
        desc.csEntry(kInferEntry);
        mpInferPass = ComputePass::create(mpDevice, desc);
    }
}
