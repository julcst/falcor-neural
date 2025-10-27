#include "SPPM.h"

#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

namespace
{
    const std::string kShaderFolder = "RenderPasses/SPPM/";
    const std::string kShaderTracePhotons = kShaderFolder + "TracePhotons.rt.slang";
    const std::string kShaderTraceQueries = kShaderFolder + "TraceQueries.rt.slang";
    const std::string kShaderBuildQueryBounds = kShaderFolder + "BuildQueryBounds.cs.slang";
    const std::string kShaderResolveQueries = kShaderFolder + "ResolveQueries.cs.slang";
    const std::string kOutputColor = "color";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SPPM>();
}

SPPM::SPPM(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

Properties SPPM::getProperties() const
{
    return {};
}

RenderPassReflection SPPM::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector
        .addOutput(kOutputColor, "SPPM radiance accumulation")
        .format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
    return reflector;
}

void SPPM::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto pOutput = renderData.getTexture(kOutputColor);
    if (!pOutput)
        return;

    if (!mpScene)
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), float4(0.f));
        return;
    }

    // Ensure lighting state and samplers are prepared before dispatching.
    prepareLightingStructure(pRenderContext);
    if (!mHasLights)
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), float4(0.f));
        return;
    }

    // When using debug views, request an accumulation reset to avoid mixing frames.
    {
        auto& dict = renderData.getDictionary();
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        if (mDebugMode != 0) flags |= RenderPassRefreshFlags::RenderOptionsChanged;
        if (mPrevDebugMode != mDebugMode)
        {
            flags |= RenderPassRefreshFlags::RenderOptionsChanged;
            mPrevDebugMode = mDebugMode;
        }
        dict[Falcor::kRenderPassRefreshFlags] = flags;
    }

    const uint2 frameDim = uint2(pOutput->getWidth(), pOutput->getHeight());
    if (frameDim.x == 0 || frameDim.y == 0)
        return;

    // Lazy allocation for accumulation texture.
    if (!mpAccumulation || mpAccumulation->getWidth() != frameDim.x || mpAccumulation->getHeight() != frameDim.y)
    {
        mpAccumulation = mpDevice->createTexture2D(
            frameDim.x,
            frameDim.y,
            ResourceFormat::RGBA32Float,
            1,
            1,
            nullptr,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
        );
        pRenderContext->clearUAV(mpAccumulation->getUAV().get(), float4(0.f));
    }

    traceQueries(pRenderContext, renderData);
    logInfo("SPPM: traceQueries() done. mQueryCount={} frameDim={}x{}", mQueryCount, mFrameDim.x, mFrameDim.y);
    // Ensure query buffers written by TraceQueries are visible to subsequent AS build and shaders.
    if (mpQueryAABBBuffer) pRenderContext->uavBarrier(mpQueryAABBBuffer.get());
    if (mpQueryBuffer) pRenderContext->uavBarrier(mpQueryBuffer.get());

    // Debug: sample a few AABBs to verify they look reasonable.
    if (mDebugMode >= 4 && mpQueryAABBBuffer && mQueryCount >= 4)
    {
        static ref<Buffer> pDbgReadback;
        if (!pDbgReadback || pDbgReadback->getSize() < mpQueryAABBBuffer->getSize())
        {
            pDbgReadback = mpDevice->createBuffer(mpQueryAABBBuffer->getSize(), ResourceBindFlags::None, MemoryType::ReadBack);
            pDbgReadback->setName("SPPM Query AABB Readback");
        }
        pRenderContext->copyBufferRegion(pDbgReadback.get(), 0, mpQueryAABBBuffer.get(), 0, std::min<uint64_t>(mpQueryAABBBuffer->getSize(), 64ull * sizeof(float)));
        pRenderContext->submit(true);
        struct RtAABB { float3 min; float3 max; };
        const RtAABB* a = reinterpret_cast<const RtAABB*>(pDbgReadback->map());
        if (a)
        {
            logInfo("SPPM Debug: AABB[0]=min({}, {}, {}) max({}, {}, {})", a[0].min.x, a[0].min.y, a[0].min.z, a[0].max.x, a[0].max.y, a[0].max.z);
            logInfo("SPPM Debug: AABB[1]=min({}, {}, {}) max({}, {}, {})", a[1].min.x, a[1].min.y, a[1].min.z, a[1].max.x, a[1].max.y, a[1].max.z);
            logInfo("SPPM Debug: AABB[2]=min({}, {}, {}) max({}, {}, {})", a[2].min.x, a[2].min.y, a[2].min.z, a[2].max.x, a[2].max.y, a[2].max.z);
            pDbgReadback->unmap();
        }
    }
    buildQueryAcceleration(pRenderContext);
    logInfo("SPPM: buildQueryAcceleration() done. BLAS={} TLAS={}", mpQueryBLAS ? "ok" : "null", mpQueryTLAS ? "ok" : "null");

    if (mQueryCount == 0 || !mpQueryTLAS)
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), float4(0.f));
        return;
    }

    if (mpQueryAccumulator)
            pRenderContext->clearUAV(mpQueryAccumulator->getUAV().get(), uint4(0));

    logInfo("SPPM: starting tracePhotonsPass()");
    if (mMixedLights)
    {
        // Dispatch emissive photons first (analyticOnly=false), then analytic photons (analyticOnly=true)
        tracePhotonsPass(pRenderContext, renderData, /*analyticOnly*/ false, /*buildAS*/ true);
        tracePhotonsPass(pRenderContext, renderData, /*analyticOnly*/ true, /*buildAS*/ true);
    }
    else
    {
        // Single pass depending on available light type.
        const bool analyticOnly = mHasAnalyticLights;
        tracePhotonsPass(pRenderContext, renderData, analyticOnly, true);
    }
    logInfo("SPPM: tracePhotonsPass() done");

    // Insert a UAV barrier to ensure preceding raytracing writes (atomics) are visible before resolve.
    if (mpQueryAccumulator) pRenderContext->uavBarrier(mpQueryAccumulator.get());
    if (mpQueryBuffer) pRenderContext->uavBarrier(mpQueryBuffer.get());
    if (mpAccumulation) pRenderContext->uavBarrier(mpAccumulation.get());
    if (mpDebugCounters) pRenderContext->uavBarrier(mpDebugCounters.get());
    // Resolve using a fullscreen graphics pass to avoid compute pipeline issues on this platform.
    logInfo("SPPM: starting resolveQueries() [FS]");
    // Read back debug counters and log (useful when output is black)
    if (mpDebugCounters)
    {
        if (!mpDebugCountersReadback || mpDebugCountersReadback->getSize() < mpDebugCounters->getSize())
        {
            mpDebugCountersReadback = mpDevice->createBuffer(mpDebugCounters->getSize(), ResourceBindFlags::None, MemoryType::ReadBack);
            mpDebugCountersReadback->setName("SPPM Debug Counters Readback");
        }
        pRenderContext->copyBufferRegion(mpDebugCountersReadback.get(), 0, mpDebugCounters.get(), 0, mpDebugCounters->getSize());
        pRenderContext->submit(true);
        const uint32_t* counters = reinterpret_cast<const uint32_t*>(mpDebugCountersReadback->map());
        if (counters)
        {
            uint32_t considered = counters[0];
            uint32_t candidates = counters[1];
            uint32_t accumulations = counters[2];
            uint32_t emitted = counters[3];
            uint32_t totalBounces = counters[4];
            logInfo("SPPM Debug: emitted={} considered={} candidates={} accumulations={} totalBounces={}", emitted, considered, candidates, accumulations, totalBounces);
            mpDebugCountersReadback->unmap();
        }
    }
    resolveQueries(pRenderContext, renderData);
    logInfo("SPPM: resolveQueries() done [FS]");

    ++mFrameCount;
}

// UI
void SPPM::renderUI(Gui::Widgets& widget)
{
    Gui::DropdownList modes = {
        {0, "Flux Resolve"},
        {1, "Query Valid"},
        {2, "Query Normals"},
        {3, "Query Diffuse"},
        {4, "Hit Count Heatmap"},
        {5, "Debug Counters (RGB)"}
    };
    uint mode = mDebugMode;
    if (widget.dropdown("Debug Mode", modes, mode)) mDebugMode = mode;

    widget.slider("Photon Radius", mPhotonRadius.x, 0.001f, 0.1f);
}

void SPPM::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) {
    // Reset Scene
    mpScene = pScene;

    if (mpScene)
    {
        if (mpScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("This render pass only supports triangles. Other types of geometry will be ignored.");
        }

        // Reset lighting state and (re)create samplers next execute().
        mpEmissiveLightSampler.reset();
        mHasLights = mHasAnalyticLights = mMixedLights = false;
    }
}

void SPPM::traceQueries(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "TraceQueries");

    auto pOutput = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutput);

    if (!mTraceQueryPass.pProgram)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderTraceQueries);
        desc.setMaxPayloadSize(sizeof(uint32_t));
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(1);

        mTraceQueryPass.pBindingTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        auto& sbt = mTraceQueryPass.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen", mpScene->getTypeConformances()));
        sbt->setMiss(0, desc.addMiss("miss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));
        }

        DefineList defines;
        defines.add(mpScene->getSceneDefines());

        mTraceQueryPass.pProgram = Program::create(mpDevice, desc, defines);
    }

    if (!mTraceQueryPass.pVars)
        mTraceQueryPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);

    // Establish frame/query sizes now, then (re)allocate buffers using reflection for correct strides.
    mFrameDim = uint2(pOutput->getWidth(), pOutput->getHeight());
    mQueryCount = mFrameDim.x * mFrameDim.y;

    if (mQueryCount == 0)
    {
        mpQueryBuffer.reset();
        mpQueryAccumulator.reset();
        mpQueryAABBBuffer.reset();
        mpQueryTLAS.reset();
        mpQueryBLAS.reset();
        return;
    }

    auto qVar = mTraceQueryPass.pVars->getRootVar();

    // Allocate PhotonQuery buffer based on shader reflection to avoid duplication.
    if (!mpQueryBuffer || mpQueryBuffer->getElementCount() != mQueryCount)
    {
        mpQueryBuffer = mpDevice->createStructuredBuffer(
            qVar["gPhotonQueries"].getType(),
            mQueryCount,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mpQueryBuffer->setName("SPPM Photon Queries");
    }

    // Allocate Query AABB buffer written by TraceQueries.
    if (!mpQueryAABBBuffer || mpQueryAABBBuffer->getElementCount() != mQueryCount)
    {
        mpQueryAABBBuffer = mpDevice->createStructuredBuffer(
            static_cast<uint32_t>(sizeof(RtAABB)),
            mQueryCount,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mpQueryAABBBuffer->setName("SPPM Query AABBs");
    }

    // Ensure accumulation buffer exists (structured float4), using an explicit 16B stride to keep things simple.
    if (!mpQueryAccumulator || mpQueryAccumulator->getElementCount() != mQueryCount)
    {
        mpQueryAccumulator = mpDevice->createStructuredBuffer(
            /* structSize = */ 16,
            mQueryCount,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mpQueryAccumulator->setName("SPPM Query Accum");
    }

    // Bind constants and UAVs.
    qVar["CB"]["gFrameDim"] = mFrameDim;
    qVar["CB"]["gFrameIndex"] = mFrameCount;
    qVar["CB"]["gQueryRadius"] = mPhotonRadius.x;
    qVar["gPhotonQueries"] = mpQueryBuffer;
    qVar["gQueryAABBs"] = mpQueryAABBBuffer;

    mpScene->raytrace(pRenderContext, mTraceQueryPass.pProgram.get(), mTraceQueryPass.pVars, uint3(mFrameDim.x, mFrameDim.y, 1));
}

void SPPM::buildQueryAcceleration(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "BuildQueryAS");

    if (mQueryCount == 0 || !mpQueryBuffer)
    {
        mpQueryTLAS.reset();
        mpQueryBLAS.reset();
        return;
    }

    // AABBs already written by TraceQueries.

    RtGeometryDesc geometryDesc = {};
    geometryDesc.type = RtGeometryType::ProcedurePrimitives;
    geometryDesc.flags = RtGeometryFlags::None;
    geometryDesc.content.proceduralAABBs.count = mQueryCount;
    geometryDesc.content.proceduralAABBs.data = mpQueryAABBBuffer->getGpuAddress();
    geometryDesc.content.proceduralAABBs.stride = sizeof(RtAABB);

    RtAccelerationStructureBuildInputs blasInputs = {};
    blasInputs.kind = RtAccelerationStructureKind::BottomLevel;
    blasInputs.flags = RtAccelerationStructureBuildFlags::PreferFastTrace;
    blasInputs.descCount = 1;
    blasInputs.geometryDescs = &geometryDesc;

    const auto blasPrebuild = RtAccelerationStructure::getPrebuildInfo(mpDevice.get(), blasInputs);

    auto ensureASBuffer = [&](ref<Buffer>& buffer, uint64_t requiredSize, ResourceBindFlags flags, const char* name)
    {
        if (!buffer || buffer->getSize() < requiredSize)
        {
            buffer = mpDevice->createBuffer(requiredSize, flags, MemoryType::DeviceLocal, nullptr);
            buffer->setName(name);
        }
    };

    ensureASBuffer(mpQueryBlasStorage, blasPrebuild.resultDataMaxSize, ResourceBindFlags::AccelerationStructure, "SPPM Query BLAS");
    ensureASBuffer(mpQueryBlasScratch, blasPrebuild.scratchDataSize, ResourceBindFlags::UnorderedAccess, "SPPM Query BLAS Scratch");

    if (!mpQueryBLAS)
    {
        RtAccelerationStructure::Desc desc;
        desc.setKind(RtAccelerationStructureKind::BottomLevel).setBuffer(mpQueryBlasStorage, 0, mpQueryBlasStorage->getSize());
        mpQueryBLAS = RtAccelerationStructure::create(mpDevice, desc);
    }

    RtAccelerationStructure::BuildDesc blasBuild = {};
    blasBuild.inputs = blasInputs;
    blasBuild.dest = mpQueryBLAS.get();
    blasBuild.source = nullptr;
    blasBuild.scratchData = mpQueryBlasScratch->getGpuAddress();

    pRenderContext->buildAccelerationStructure(blasBuild, 0, nullptr);

    RtInstanceDesc instanceDesc = {};
    // Set identity transform (3x4 row-major)
    instanceDesc.transform[0][0] = 1.0f; instanceDesc.transform[0][1] = 0.0f; instanceDesc.transform[0][2] = 0.0f; instanceDesc.transform[0][3] = 0.0f;
    instanceDesc.transform[1][0] = 0.0f; instanceDesc.transform[1][1] = 1.0f; instanceDesc.transform[1][2] = 0.0f; instanceDesc.transform[1][3] = 0.0f;
    instanceDesc.transform[2][0] = 0.0f; instanceDesc.transform[2][1] = 0.0f; instanceDesc.transform[2][2] = 1.0f; instanceDesc.transform[2][3] = 0.0f;
    instanceDesc.instanceID = 0;
    instanceDesc.instanceMask = 0xFF;
    instanceDesc.instanceContributionToHitGroupIndex = mpScene ? mpScene->getGeometryCount() : 0;
    instanceDesc.flags = RtGeometryInstanceFlags::ForceOpaque;
    instanceDesc.accelerationStructure = mpQueryBLAS->getGpuAddress();

    if (!mpQueryInstanceBuffer)
    {
        mpQueryInstanceBuffer = mpDevice->createBuffer(sizeof(RtInstanceDesc), ResourceBindFlags::AccelerationStructure, MemoryType::Upload, nullptr);
        mpQueryInstanceBuffer->setName("SPPM Query TLAS Instances");
    }
    mpQueryInstanceBuffer->setBlob(&instanceDesc, 0, sizeof(instanceDesc));

    RtAccelerationStructureBuildInputs tlasInputs = {};
    tlasInputs.kind = RtAccelerationStructureKind::TopLevel;
    tlasInputs.flags = RtAccelerationStructureBuildFlags::PreferFastTrace;
    tlasInputs.descCount = 1;
    tlasInputs.instanceDescs = mpQueryInstanceBuffer->getGpuAddress();

    const auto tlasPrebuild = RtAccelerationStructure::getPrebuildInfo(mpDevice.get(), tlasInputs);

    ensureASBuffer(mpQueryTlasStorage, tlasPrebuild.resultDataMaxSize, ResourceBindFlags::AccelerationStructure, "SPPM Query TLAS");
    ensureASBuffer(mpQueryTlasScratch, tlasPrebuild.scratchDataSize, ResourceBindFlags::UnorderedAccess, "SPPM Query TLAS Scratch");

    if (!mpQueryTLAS)
    {
        RtAccelerationStructure::Desc desc;
        desc.setKind(RtAccelerationStructureKind::TopLevel).setBuffer(mpQueryTlasStorage, 0, mpQueryTlasStorage->getSize());
        mpQueryTLAS = RtAccelerationStructure::create(mpDevice, desc);
    }

    RtAccelerationStructure::BuildDesc tlasBuild = {};
    tlasBuild.inputs = tlasInputs;
    tlasBuild.dest = mpQueryTLAS.get();
    tlasBuild.source = nullptr;
    tlasBuild.scratchData = mpQueryTlasScratch->getGpuAddress();

    pRenderContext->buildAccelerationStructure(tlasBuild, 0, nullptr);
}

void SPPM::resolveQueries(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "ResolveQueriesFS");

    if (mQueryCount == 0 || !mpQueryAccumulator)
        return;

    auto pOutput = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutput);

    if (!mpResolveFullScreen)
    {
        ProgramDesc desc;
        desc.addShaderLibrary("RenderPasses/SPPM/ResolveQueries.ps.slang");
        desc.psEntry("psMain");
        mpResolveFullScreen = FullScreenPass::create(mpDevice, desc);
    }

    auto var = mpResolveFullScreen->getRootVar();
    var["CB"]["gFrameDim"] = mFrameDim;
    var["CB"]["gFrameIndex"] = mFrameCount;
    var["CB"]["gDebugMode"] = mDebugMode;
    var["CB"]["gAccumScale"] = 65536.0f;
    var["gQueryAccumulationBuf"] = mpQueryAccumulator;
    var["gPhotonQueries"] = mpQueryBuffer;
    if (mpDebugCounters) var["gDebugCounters"] = mpDebugCounters;

    // Draw to output
    std::vector<ref<Texture>> colors = { pOutput };
    auto pFbo = Fbo::create(mpDevice, colors);
    mpResolveFullScreen->execute(pRenderContext, pFbo);
}

void SPPM::tracePhotonsPass(RenderContext* pRenderContext, const RenderData& renderData,  bool analyticOnly,  bool buildAS)
{
    FALCOR_PROFILE(pRenderContext, "TracePhotons");

    const uint32_t sceneGeometryCount = mpScene ? mpScene->getGeometryCount() : 0;
    const uint32_t queryGeometryIndex = sceneGeometryCount; // Extra slot reserved for photon queries.

    // Init/update emissive sampler and shader
    if (mpEmissiveLightSampler)
    {
        mpEmissiveLightSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
    }
    if (!mTracePhotonPass.pProgram)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderTracePhotons);
        desc.setMaxPayloadSize(sizeof(float) * 12);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(2);

        // Use a binding table matching the scene's geometry count. Query AS will be traversed using inline RayQuery, so no extra SBT entries are needed.
        mTracePhotonPass.pBindingTable = RtBindingTable::create(1, 1, sceneGeometryCount);
        auto& sbt = mTracePhotonPass.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen", mpScene->getTypeConformances()));
        sbt->setMiss(0, desc.addMiss("miss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));
        }
        DefineList defines;
        // Always compile emissive light path; we'll choose the branch at runtime.
        defines.add("USE_EMISSIVE_LIGHTS", "1");
        defines.add(mpScene->getSceneDefines());
        if (mpEmissiveLightSampler)
            defines.add(mpEmissiveLightSampler->getDefines());


        mTracePhotonPass.pProgram = Program::create(mpDevice, desc, defines);
    }
    // Late define changes would require recompile; keep defines in creation above.

    // Program Vars
    if (!mTracePhotonPass.pVars)
        mTracePhotonPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);

    FALCOR_ASSERT(mTracePhotonPass.pVars);
    auto var = mTracePhotonPass.pVars->getRootVar();

    var["CB"]["gQueryCount"] = mQueryCount;
    if (mpQueryTLAS)
        var["gQueryAS"].setAccelerationStructure(mpQueryTLAS);
    if (mpQueryBuffer)
        var["gPhotonQueries"] = mpQueryBuffer;
    if (mpQueryAccumulator)
        var["gQueryAccumulation"] = mpQueryAccumulator;

    // Allocate/clear debug counters and bind.
    if (!mpDebugCounters)
    {
        mpDebugCounters = mpDevice->createStructuredBuffer(
            /* structSize = */ sizeof(uint32_t),
            /* elementCount = */ 8,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mpDebugCounters->setName("SPPM Debug Counters");
    }
    // Clear counters each frame.
    pRenderContext->clearUAV(mpDebugCounters->getUAV().get(), uint4(0));
    var["gDebugCounters"] = mpDebugCounters;

    // Handle shader dimension
    uint dispatchedPhotons = mNumDispatchedPhotons;
    if (mMixedLights)
    {
        float dispatchedF = float(dispatchedPhotons);
        dispatchedF *= analyticOnly ? mPhotonAnalyticRatio : 1.f - mPhotonAnalyticRatio;
        dispatchedPhotons = uint(dispatchedF);
    }
    uint shaderDispatchDim = static_cast<uint>(std::floor(std::sqrt(static_cast<float>(dispatchedPhotons))));
    shaderDispatchDim = std::max(32u, shaderDispatchDim);
    const uint actualPhotons = shaderDispatchDim * shaderDispatchDim;
    logInfo("SPPM: tracing {} photons (dispatch {}x{}, requested={}, analyticOnly={})",
        actualPhotons, shaderDispatchDim, shaderDispatchDim, dispatchedPhotons, analyticOnly);

    // Constant Buffer
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPhotonRadius"] = mPhotonRadius;
    var["CB"]["gMaxBounces"] = mPhotonMaxBounces;
    var["CB"]["gGlobalRejectionProb"] = mGlobalPhotonRejection;
    var["CB"]["gUseAnalyticLights"] = analyticOnly;
    var["CB"]["gDispatchDimension"] = shaderDispatchDim;
    var["CB"]["gAccumScale"] = 65536.0f;
    // Enable debug probing when using debug vis modes (>=4): doubleDir and larger TMax.
    // In debug (>=4), probe in more directions and with long TMax
    uint debugFlags = (mDebugMode >= 4) ? (0x1u | 0x2u | 0x4u) : 0u;
    var["CB"]["gDebugFlags"] = debugFlags;

    // Structures
    if (mpEmissiveLightSampler)
        mpEmissiveLightSampler->bindShaderData(var["Light"]["gEmissiveSampler"]);

    // Dispatch raytracing shader
    mpScene->raytrace(pRenderContext, mTracePhotonPass.pProgram.get(), mTracePhotonPass.pVars, uint3(shaderDispatchDim, shaderDispatchDim, 1));
}

void SPPM::prepareLightingStructure(RenderContext* pRenderContext)
{
    if (!mpScene) { mHasLights = mHasAnalyticLights = mMixedLights = false; return; }

    // Ensure emissive light collection is up to date on CPU side for correct flags/info.
    if (auto pLights = mpScene->getILightCollection(pRenderContext))
    {
        pLights->prepareSyncCPUData(pRenderContext);
    }

    const bool emissiveUsed = mpScene->useEmissiveLights();
    const bool analyticUsed = mpScene->useAnalyticLights();
    mHasAnalyticLights = analyticUsed;
    mHasLights = emissiveUsed || analyticUsed;
    mMixedLights = emissiveUsed && analyticUsed;

    // Create/update emissive light sampler if emissive lights are present.
    if (emissiveUsed)
    {
        if (!mpEmissiveLightSampler)
        {
            auto pLights = mpScene->getILightCollection(pRenderContext);
            if (pLights)
            {
                mpEmissiveLightSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, pLights);
            }
        }
        if (mpEmissiveLightSampler)
        {
            mpEmissiveLightSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
        }
    }
    else
    {
        mpEmissiveLightSampler.reset();
    }
}

void SPPM::RayTraceProgramHelper::initProgramVars(const ref<Device>& pDevice, const ref<Scene>& pScene, const ref<SampleGenerator>& pSampleGenerator)
{
    FALCOR_ASSERT(pProgram);

    // Configure program.
    pProgram->addDefines(pSampleGenerator->getDefines());
    pProgram->setTypeConformances(pScene->getTypeConformances());
    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    pVars = RtProgramVars::create(pDevice, pProgram, pBindingTable);

    // Bind utility classes into shared data.
    auto var = pVars->getRootVar();
    pSampleGenerator->bindShaderData(var);
}
