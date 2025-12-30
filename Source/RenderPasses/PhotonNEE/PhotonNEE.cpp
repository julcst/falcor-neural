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
#include "PhotonNEE.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "../TraceQueries/Query.slang"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, PhotonNEE>();
}

namespace
{
    const char kShaderFile[] = "RenderPasses/PhotonNEE/PhotonNEE.rt.slang";

    const char kInputQueries[] = "queries";
    const char kOutputRadiance[] = "output";

    const char kMaxBounces[] = "maxBounces";
    const char kRussianRouletteWeight[] = "russianRouletteWeight";
    const char kUseRussianRoulette[] = "useRussianRoulette";

    // Ray tracing settings
    const uint32_t kMaxPayloadSizeBytes = 72u; // Estimate
    const uint32_t kMaxRecursionDepth = 2u; // 1 for shadow ray?
}

PhotonNEE::PhotonNEE(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    setProperties(props);
}

void PhotonNEE::setProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMaxBounces) mMaxBounces = value;
        else if (key == kRussianRouletteWeight) mRussianRouletteWeight = value;
        else if (key == kUseRussianRoulette) mUseRussianRoulette = value;
        else logWarning("{} - Unrecognized property '{}'", getClassName(), key);
    }
}

Properties PhotonNEE::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kRussianRouletteWeight] = mRussianRouletteWeight;
    props[kUseRussianRoulette] = mUseRussianRoulette;
    return props;
}

RenderPassReflection PhotonNEE::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputQueries, "Query buffer")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    reflector.addOutput(kOutputRadiance, "Radiance output")
        .rawBuffer(mQueryCount * sizeof(float3))
        .bindFlags(ResourceBindFlags::UnorderedAccess);
    return reflector;
}

void PhotonNEE::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    const uint queryCount = compileData.connectedResources.getField(kInputQueries)->getWidth() / sizeof(Query);
    if (queryCount != mQueryCount)
    {
        mQueryCount = queryCount;
        FALCOR_CHECK(false, "Force recompile");
    }
}

void PhotonNEE::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mTracer.pProgram.reset();
    mTracer.pBindingTable.reset();
    mTracer.pVars.reset();
    mFrameCount = 0;
    mpPhotonSampler.reset();

    mpScene = pScene;

    if (mpScene)
    {
        mpPhotonSampler = PhotonSampler::create(pRenderContext, mpScene);

        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGenWarp"));
        sbt->setMiss(0, desc.addMiss("miss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));
            sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit"));
        }

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        mTracer.pProgram->addDefines(mpPhotonSampler->getDefines());
    }
}

void PhotonNEE::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    mpPhotonSampler->update(pRenderContext);

    // Specialize program
    mTracer.pProgram->addDefines(mpPhotonSampler->getDefines());
    mTracer.pProgram->addDefine("USE_RUSSIAN_ROULETTE", mUseRussianRoulette ? "1" : "0");

    if (!mTracer.pVars) prepareVars();

    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gQueryCount"] = mQueryCount;
    var["CB"]["gMaxBounces"] = mMaxBounces;
    var["CB"]["gRussianRouletteWeight"] = mRussianRouletteWeight;
    
    // Bind resources
    var["gQueries"] = renderData.getBuffer(kInputQueries);
    var["gOutput"] = renderData.getBuffer(kOutputRadiance);
    
    mpPhotonSampler->bindShaderData(var["gLights"]);

    // Trace
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(mQueryCount, 1, 1));

    mFrameCount++;
}

void PhotonNEE::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}

void PhotonNEE::renderUI(Gui::Widgets& widget)
{
    widget.var("Max Bounces", mMaxBounces, 1u, 16u);
    widget.var("Russian Roulette Weight", mRussianRouletteWeight, 0.1f, 10.f);
    widget.checkbox("Use Russian Roulette", mUseRussianRoulette);
}
