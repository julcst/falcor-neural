This is a simple agent implementation for the Falcor + Slang.

## Documentation

For a Documentation of Slang, see [here](https://shader-slang.org/docs/llms.txt) and for full documentation [here](https://shader-slang.org/docs/llms_full.txt).

Falcor is a rendering framework with a render graph implementation, so you only write render passes with inputs, outputs and intermediate buffers. The render graph will take care of the execution order and resource management. A render pass looks e.g. like this:

```cpp
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
constexpr uint32_t kOptimizeDispatchThreads = (std::min(1u<<19u, MLPConfig::kEncodingParamElementCount) + 255u) / 256u * 256u;
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

    ...
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

    ...
}
```

See also: docs/usage/render-passes.md

To have a buffer size dependent e.g. on an input or output buffer, you have to overload compile:
```cpp
void AccumulatePhotonsRTX::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    for (auto i = 0; i < compileData.connectedResources.getFieldCount(); ++i) {
        const auto& field = compileData.connectedResources.getField(i);
        logInfo("Connected resource: {} (width={})", field->getName(), field->getWidth());
    }
    const auto queryCount = compileData.connectedResources.getField(kQueryBuffer)->getWidth() / sizeof(Query);
    const auto photonHitCount = compileData.connectedResources.getField(kPhotonBuffer)->getWidth() / sizeof(PhotonHit);
    logInfo("queryCount={}, photonHitCount={}", queryCount, photonHitCount);
    if (mQueryCount != queryCount || mPhotonHitCount != photonHitCount) {
        mQueryCount = queryCount;
        mPhotonHitCount = photonHitCount;
        FALCOR_CHECK(false, "Recompile with new buffer sizes"); // Force retry of reflect
    }
}
```

For coding conventions look at docs/development/coding-conventions.md and docs/development/error-handling.md.

It is crucial to list all Shaders in a CMakeLists.txt file, otherwise they won't be copied to the output directory and shader compilation will fail during runtime. You can find an example in Source/RenderPasses/SlangMLP/CMakeLists.txt.

For Slang/C++ interop, you can define a struct in a .slang file and include it in both the .slang and .cpp files:

```cpp
#pragma once

#include "Utils/HostDeviceShared.slangh"
#include "Utils/Math/MathConstants.slangh"

BEGIN_NAMESPACE_FALCOR

struct NRCInput
{
    float3 position;
    float roughness;
    float3 wo;
    float3 wn;
    float3 diffuse;
    float3 specular;

#ifndef HOST_CODE
    __init()
    {
        position = {0};
        wo = float3(1, 0, 0);
        wn = float3(1, 0, 0);
        roughness = 0.5;
        diffuse = {0};
        specular = {0};
    }

    float3 weight()
    {
        return (diffuse + specular) + 1e-2f;
    }
#endif
};

static const uint NRC_INPUT_SIZE = sizeof(NRCInput) / sizeof(float);

struct NRCOutput
{
    float3 radiance;
};

static const uint NRC_OUTPUT_SIZE = sizeof(NRCOutput) / sizeof(float);

static const uint NRC_BATCH_SIZE_GRANULARITY = 256;

END_NAMESPACE_FALCOR
```

To build a render graph, Python scripts are used like this:

```python
from falcor import *


def render_graph_PT():
    g = RenderGraph("PT")
    g.createPass("accum", "AccumulatePass", {"precisionMode": "Double"})
    g.createPass("pt", "PathTracer", {"maxDiffuseBounces": 8, "maxSpecularBounces": 8})
    g.createPass("vbuff", "VBufferRT", {})

    g.addEdge("vbuff.vbuffer", "pt.vbuffer")
    g.addEdge("vbuff.viewW", "pt.viewW")
    g.addEdge("vbuff.mvec", "pt.mvec")
    g.addEdge("pt.color", "accum.input")
    g.markOutput("accum.output")
    return g

def _register(g):
    try:
        m.addGraph(g)
    except NameError:
        pass

m.loadScene("../scenes/cornell_box_bunny.pyscene")
_register(render_graph_PT())
```

To inspect frames, you can add frames to the frameCapture:

```python
m.frameCapture.addFrames(m.activeGraph, range(0, 1000, 100))
```

To export timings:

```python
m.timingCapture.captureFrameTime("timings.csv")
```
Or for in depth profiling:

```python
m.profiler.enabled = True
m.profiler.start_capture()
for frame in range(256):
    m.renderFrame()
capture = m.profiler.end_capture()
m.profiler.enabled = False

print(f"Mean frame time: {capture['events']['/onFrameRender/RenderGraphExe::execute()/SlangMLP/SlangMLP/gpu_time']['stats']['mean']} ms")
```

## Building

We only use CMake to build the application. This is necessary for changes of the source files but also for changes of the shader files, as CMake will trigger a shader copy to the output directory. Prefer The CMake Build Tools of VSCode for building and debugging.

To run the application, we run e.g. build/linux-gcc/bin/Debug/Mogwai --script scripts/SlangMLP.py, shader compilation failures will only showup during runtime so it is crucial that you run the application when you made changes to the shader files. Mogwai is a GUI application, it requires user interaction to quit the application, even if errors are thrown. You can run it headless with --headless, but it will still require manual termination through the terminal, e.g. with Ctrl+\\.

