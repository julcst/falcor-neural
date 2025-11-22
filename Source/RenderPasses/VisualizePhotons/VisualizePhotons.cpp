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
#include "VisualizePhotons.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "../TracePhotons/Structs.slang"

namespace
{
    const char kShaderFile[] = "RenderPasses/VisualizePhotons/VisualizePhotons.cs.slang";
    const std::string kPhotonBuffer = "photons";
    const std::string kCounterBuffer = "counters";
    const std::string kOutputColor = "dst";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VisualizePhotons>();
}

VisualizePhotons::VisualizePhotons(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {}

Properties VisualizePhotons::getProperties() const
{
    return {};
}

void VisualizePhotons::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}

RenderPassReflection VisualizePhotons::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    reflector.addInput(kPhotonBuffer, "Traced photons")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    
    reflector.addInput(kCounterBuffer, "Photon Counters")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

    reflector.addOutput(kOutputColor, "Photon visualization")
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
        .format(ResourceFormat::RGBA32Float)
        .texture2D(0, 0);

    return reflector;
}

void VisualizePhotons::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "VisualizePhotons");

    auto pPhotonBuffer = renderData.getResource(kPhotonBuffer)->asBuffer();
    FALCOR_ASSERT(pPhotonBuffer);
    auto pOutput = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutput);

    pRenderContext->uavBarrier(pPhotonBuffer.get());
    auto mean = float3(0.f);
    auto minF = float3(INFINITY);
    auto maxF = float3(-INFINITY);
    for (const auto photon : pPhotonBuffer->getElements<PhotonHit>())
    {
        mean += photon.flux;
        minF = min(minF, photon.flux);
        maxF = max(maxF, photon.flux);
    }
    mean /= float(pPhotonBuffer->getElementCount());
    logInfo("mean=({}, {}, {}) min=({}, {}, {}) max=({}, {}, {})\n", mean.x, mean.y, mean.z, minF.x, minF.y, minF.z, maxF.x, maxF.y, maxF.z);

    pRenderContext->clearUAV(pOutput->getUAV().get(), float4(0.f));

    if (!mpScene || !pPhotonBuffer)
    {
        return;
    }

    preparePass();
    auto var = mpVisualizePass->getRootVar();
    var["gPhotonHits"] = pPhotonBuffer;
    var["gOutput"] = pOutput;

    const auto frameDim = uint2(pOutput->getWidth(), pOutput->getHeight());
    var["CB"]["gFrameDim"] = frameDim;
    var["CB"]["gPointSizePx"] = mPointSize;

    const auto* pCamera = mpScene->getCamera().get();
    FALCOR_ASSERT(pCamera);
    const float4x4 viewProj = pCamera->getViewProjMatrix();
    var["CB"]["gViewProj"] = viewProj;

    auto pCounterBuffer = renderData.getResource(kCounterBuffer)->asBuffer();
    FALCOR_ASSERT(pCounterBuffer);
    var["gCounters"] = pCounterBuffer;
    // TODO: Use indirect dispatch to avoid CPU-GPU sync
    pRenderContext->uavBarrier(pCounterBuffer.get());
    const auto counters = pCounterBuffer->getElement<PhotonCounters>(0u);

    mpVisualizePass->execute(pRenderContext, counters.PhotonStores, 1, 1);
}

void VisualizePhotons::renderUI(Gui::Widgets& widget)
{
    widget.var("Point size", mPointSize, 1u, 32u);
}

void VisualizePhotons::preparePass()
{
    if (mpVisualizePass)
        return;

    ProgramDesc desc;
    desc.addShaderLibrary(kShaderFile);
    desc.csEntry("main");

    mpVisualizePass = ComputePass::create(mpDevice, desc);
}
