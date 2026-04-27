#include "SlangMLP.h"
#include "Config.slang"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{
const std::string kTrainShaderFile = "RenderPasses/SlangMLP/Training.cs.slang";
const std::string kOptimizeShaderFile = "RenderPasses/SlangMLP/Optimization.cs.slang";
const std::string kInferShaderFile = "RenderPasses/SlangMLP/Inference.cs.slang";
const std::string kTrainEntry = "trainMain";
const std::string kOptimizeEntry = "optimizeMain";
const std::string kResetEntry = "resetMain";
const std::string kInferEntry = "inferMain";

const std::string kInput = "src";
const std::string kOutput = "dst";
const std::string kParams = "params";
const std::string kParamGrads = "paramGrads";
const std::string kMoments1 = "moments1";
const std::string kMoments2 = "moments2";
const std::string kEncodingParams = "encodingParams";
const std::string kEncodingParamGrads = "encodingParamGrads";
const std::string kEncodingMoments1 = "encodingMoments1";
const std::string kEncodingMoments2 = "encodingMoments2";

const std::string kTrainSteps = "trainSteps";
const std::string kBatchSize = "batchSize";
const std::string kLearningRate = "learningRate";

// Keep enough groups to saturate the GPU while avoiding very large dispatch overhead.
constexpr uint32_t kOptimizeDispatchThreads = (std::max(MLPConfig::kParamElementCount, MLPConfig::kEncodingParamElementCount) + 255u) / 256u * 256u;
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
    widget.var("Train steps", mTrainSteps, 0u, 4096u);
    widget.var("Batch size", mBatchSize, 1u, 1u << 20u, 128u);
    widget.var("Learning rate", mLearningRate, 0.0f, 1e-2f);
    if (widget.button("Reset")) mReset = true;
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
    const auto pEncodingParams = renderData.getBuffer(kEncodingParams);
    const auto pEncodingParamGrads = renderData.getBuffer(kEncodingParamGrads);
    const auto pEncodingMoments1 = renderData.getBuffer(kEncodingMoments1);
    const auto pEncodingMoments2 = renderData.getBuffer(kEncodingMoments2);
    FALCOR_ASSERT(pInput && pOutput && pParams && pParamGrads && pMoments1 && pMoments2 && pEncodingParams && pEncodingParamGrads && pEncodingMoments1 && pEncodingMoments2);

    createPasses();

    if (mReset) {
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

    for (uint32_t i = 0; i < mTrainSteps; ++i)
    {
        auto var = mpTrainPass->getRootVar();
        var["gInput"] = pInput;
        var["gParams"] = pParams;
        var["gParamGrads"] = pParamGrads;
        var["gEncodingParams"] = pEncodingParams;
        var["gEncodingParamGrads"] = pEncodingParamGrads;

        var["CB"]["gFrameDim"] = uint2(pInput->getWidth(), pInput->getHeight());
        var["CB"]["gFrameIndex"] = mFrameIndex;
        var["CB"]["gBatchSize"] = mBatchSize;
        var["CB"]["gCurrentStep"] = mOptimizeStep;

        mpTrainPass->execute(pRenderContext, mBatchSize, 1u, 1u);

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

    {
        auto var = mpInferPass->getRootVar();
        var["gParams"] = pParams;
        var["gEncodingParams"] = pEncodingParams;
        var["gOutput"] = pOutput;
        var["CB"]["gFrameDim"] = uint2(pOutput->getWidth(), pOutput->getHeight());

        mpInferPass->execute(pRenderContext, pOutput->getWidth(), pOutput->getHeight(), 1u);
    }
    ++mFrameIndex;
}

void SlangMLP::createPasses()
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
