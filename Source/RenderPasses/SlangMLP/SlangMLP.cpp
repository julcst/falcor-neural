#include "SlangMLP.h"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{
const char kShaderFile[] = "RenderPasses/SlangMLP/SlangMLP.cs.slang";
const char kTrainEntry[] = "trainMain";
const char kInferEntry[] = "inferMain";

const std::string kInput = "src";
const std::string kOutput = "dst";
const std::string kParams = "params";

const char kTrainSteps[] = "trainSteps";
const char kLearningRate[] = "learningRate";

constexpr uint32_t kHiddenSize = 8;
constexpr uint32_t kParamCount = 2 * kHiddenSize + kHiddenSize + 3 * kHiddenSize + 3;
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
        else if (key == kLearningRate)
            mLearningRate = value;
        else
            logWarning("{} - Unrecognized property '{}'", getClassName(), key);
    }

    mTrainSteps = std::max(1u, mTrainSteps);
    mLearningRate = std::max(1e-6f, mLearningRate);
}

void SlangMLP::renderUI(Gui::Widgets& widget)
{
    widget.var("Train steps", mTrainSteps, 1u, 4096u);
    widget.var("Learning rate", mLearningRate, 1e-5f, 1.0f, 1e-4f, true);
    if (widget.button("Reset"))
        mReset = true;
}

Properties SlangMLP::getProperties() const
{
    Properties props;
    props[kTrainSteps] = mTrainSteps;
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
        .rawBuffer(kParamCount * sizeof(float))
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
    FALCOR_ASSERT(pInput && pOutput && pParams);

    createPasses();

    {
        auto var = mpTrainPass->getRootVar();
        var["gInput"] = pInput;
        var["gOutput"] = pOutput;
        var["gParams"] = pParams;

        var["CB"]["gFrameDim"] = uint2(pInput->getWidth(), pInput->getHeight());
        var["CB"]["gFrameIndex"] = mFrameIndex;
        var["CB"]["gTrainSteps"] = mTrainSteps;
        var["CB"]["gLearningRate"] = mLearningRate;
        var["CB"]["gReset"] = mReset ? 1u : 0u;

        mpTrainPass->execute(pRenderContext, 1u, 1u, 1u);
    }

    {
        auto var = mpInferPass->getRootVar();
        var["gInput"] = pInput;
        var["gParams"] = pParams;
        var["gOutput"] = pOutput;
        var["CB"]["gFrameDim"] = uint2(pOutput->getWidth(), pOutput->getHeight());
        mpInferPass->execute(pRenderContext, pOutput->getWidth(), pOutput->getHeight(), 1u);
    }

    mReset = false;
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

    if (!mpInferPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderFile);
        desc.csEntry(kInferEntry);
        mpInferPass = ComputePass::create(mpDevice, desc);
    }
}
