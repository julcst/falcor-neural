#include "SPPM.h"

#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

namespace
{
    const std::string kShaderFolder = "RenderPasses/SPPM/";
    const std::string kShaderTracePhotons = kShaderFolder + "TracePhotons.rt.slang";
    const std::string kShaderTraceQueries = kShaderFolder + "TraceQueries.rt.slang";
    const std::string kShaderBuildQueryBounds = kShaderFolder + "BuildQueryBounds.cs.slang";
    const std::string kShaderIntersectPhotons = kShaderFolder + "IntersectPhotons.rt.slang";
    const std::string kShaderBuildQueryGrid = kShaderFolder + "BuildQueryGrid.cs.slang";
    const std::string kShaderAccumulateByGrid = kShaderFolder + "AccumulateByGrid.cs.slang";
    const std::string kShaderResolvePS = kShaderFolder + "SPPMResolve.ps.slang";
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

    // No separate accumulation render target is used here; the fullscreen resolve writes directly to the output.

    // Allocate/clear debug counters before any pass uses them.
    if (!mpDebugCounters)
    {
        mpDebugCounters = mpDevice->createStructuredBuffer(
            /* structSize = */ 64u,   // sizeof(SPPMCounters)
            /* elementCount = */ 1u,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
        mpDebugCounters->setName("SPPM Debug Counters");
    }
    pRenderContext->clearUAV(mpDebugCounters->getUAV().get(), uint4(0));

    logInfo("SPPM: photon radius = {}", mPhotonRadius.x);

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
    // Reset photon hit counter before tracing photons.
    if (!mpPhotonHitCounter)
    {
        mpPhotonHitCounter = mpDevice->createStructuredBuffer(
            sizeof(uint32_t), 1,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal, nullptr, false);
        mpPhotonHitCounter->setName("SPPM PhotonHit Counter");
    }
    // Clear to zero
    pRenderContext->clearUAV(mpPhotonHitCounter->getUAV().get(), uint4(0));

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

    // Ensure photon hit data is visible before readback/intersect pass.
    if (mpPhotonHitCounter) pRenderContext->uavBarrier(mpPhotonHitCounter.get());
    if (mpPhotonHits) pRenderContext->uavBarrier(mpPhotonHits.get());

    // Read back photon hit count.
    uint photonHitCount = 0;
    if (mpPhotonHitCounter)
    {
        if (!mpPhotonHitCounterReadback || mpPhotonHitCounterReadback->getSize() < sizeof(uint32_t))
        {
            mpPhotonHitCounterReadback = mpDevice->createBuffer(sizeof(uint32_t), ResourceBindFlags::None, MemoryType::ReadBack);
            mpPhotonHitCounterReadback->setName("SPPM PhotonHit Counter Readback");
        }
        pRenderContext->copyBufferRegion(mpPhotonHitCounterReadback.get(), 0, mpPhotonHitCounter.get(), 0, sizeof(uint32_t));
        pRenderContext->submit(true);
        const uint32_t* pCount = reinterpret_cast<const uint32_t*>(mpPhotonHitCounterReadback->map());
        if (pCount)
        {
            photonHitCount = *pCount;
            mpPhotonHitCounterReadback->unmap();
        }
        // Clamp to capacity if we overflowed.
        if (mpPhotonHits)
        {
            uint32_t capacity = (uint32_t)mpPhotonHits->getElementCount();
            if (photonHitCount > capacity)
            {
                logWarning("SPPM: photonHitCount {} exceeds capacity {}. Clamping.", photonHitCount, capacity);
                photonHitCount = capacity;
            }
        }
        logInfo("SPPM: photonHitCount={}", photonHitCount);
    }

    if (photonHitCount > 0)
    {
        if (mUseRTXAccumulation)
        {
            // RTX any-hit accumulation path for comparison
            intersectPhotonsPass(pRenderContext, photonHitCount);
        }
        else
        {
            // Build a 3D uniform grid over queries and accumulate photon contributions via neighbor search.
            buildQueryGrid(pRenderContext);
            // Ensure grid heads/next are visible before accumulation
            if (mpGridHeads) pRenderContext->uavBarrier(mpGridHeads.get());
            if (mpGridNext) pRenderContext->uavBarrier(mpGridNext.get());
            accumulateByGrid(pRenderContext, photonHitCount);
        }
        // Ensure debug counters are visible before readback/resolve
        if (mpDebugCounters) pRenderContext->uavBarrier(mpDebugCounters.get());
        if (mpQueryAccumulator) pRenderContext->uavBarrier(mpQueryAccumulator.get());
    }

    // Insert a UAV barrier to ensure preceding raytracing writes (atomics) are visible before resolve.
    if (mpQueryAccumulator) pRenderContext->uavBarrier(mpQueryAccumulator.get());
    if (mpQueryBuffer) pRenderContext->uavBarrier(mpQueryBuffer.get());
    // (no mpAccumulation)
    if (mpDebugCounters) pRenderContext->uavBarrier(mpDebugCounters.get());

    // Debug: sample and validate a slice of the accumulation buffer before resolve.
    // This helps detect binding/visibility issues when the resolve shows black.
    if (mpQueryAccumulator && mQueryCount > 0)
    {
        static ref<Buffer> sAccumReadback;
        // Read back up to 1024 entries to keep it cheap.
        const uint64_t maxSample = std::min<uint64_t>(mQueryCount, 1024);
        const uint64_t bytes = maxSample * sizeof(uint4);
        if (!sAccumReadback || sAccumReadback->getSize() < bytes)
        {
            sAccumReadback = mpDevice->createBuffer(bytes, ResourceBindFlags::None, MemoryType::ReadBack);
            sAccumReadback->setName("SPPM Accum Readback");
        }
        // Ensure all atomics to gQueryAccumulation are visible, then copy a slice.
        pRenderContext->uavBarrier(mpQueryAccumulator.get());
        pRenderContext->copyBufferRegion(sAccumReadback.get(), 0, mpQueryAccumulator.get(), 0, bytes);
        pRenderContext->submit(true);

        const uint4* acc = reinterpret_cast<const uint4*>(sAccumReadback->map());
        if (acc)
        {
            uint nonZeroW = 0;
            uint firstIdx = UINT32_MAX;
            uint4 firstVal = uint4(0);
            uint maxW = 0;
            uint maxIdx = 0;
            for (uint i = 0; i < (uint)maxSample; ++i)
            {
                uint w = acc[i].w;
                if (w > 0)
                {
                    ++nonZeroW;
                    if (firstIdx == UINT32_MAX) { firstIdx = i; firstVal = acc[i]; }
                    if (w > maxW) { maxW = w; maxIdx = i; }
                }
            }
            if (firstIdx != UINT32_MAX)
            {
                logInfo("SPPM AccumCheck: sampled={} of {}, nonZeroW={} (first idx {} -> {{x:{}, y:{}, z:{}, w:{}}}, maxW at {} -> {})",
                        (uint)maxSample, mQueryCount, nonZeroW, firstIdx, firstVal.x, firstVal.y, firstVal.z, firstVal.w, maxIdx, maxW);
            }
            else
            {
                logWarning("SPPM AccumCheck: sampled={} of {}, all zeros in acc.w", (uint)maxSample, mQueryCount);
            }
            sAccumReadback->unmap();
        }
    }
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
        const SPPMCounters* counters = reinterpret_cast<const SPPMCounters*>(mpDebugCountersReadback->map());
        if (counters)
        {
            // Snapshot for UI
            mLastCounters = *counters;
            logInfo(
                "SPPM Debug: emitted={} considered={} candidates={} accumulations={} totalBounces={} intersectRaygen={} intersectorCalls={} validQueries={} gridNonEmptyCells={} binnedQueries={} photonStores={}",
                counters->PhotonsEmitted, counters->PhotonsConsidered, counters->Candidates, counters->Accumulations, counters->TotalBounces,
                counters->IntersectRaygen, counters->IntersectorCalls, counters->ValidQueries, counters->GridNonEmptyCells, counters->GridBinnedQueries, counters->PhotonStores
            );
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
        {5, "Was Hit"},
    };
    uint mode = mDebugMode;
    if (widget.dropdown("Debug Mode", modes, mode)) mDebugMode = mode;

    widget.slider("Photon Radius", mPhotonRadius.x, 0.001f, 0.1f);

    widget.checkbox("Use RTX Accumulation (compare)", mUseRTXAccumulation);

    // Show counters in UI (snapshot from last frame)
    widget.text("Counters (last frame):");
    widget.indent(16.0f);
    widget.text(fmt::format("emitted:\t{}", mLastCounters.PhotonsEmitted));
    widget.text(fmt::format("considered:\t{}", mLastCounters.PhotonsConsidered));
    widget.text(fmt::format("candidates:\t{}", mLastCounters.Candidates));
    widget.text(fmt::format("accum:\t\t{}", mLastCounters.Accumulations));
    widget.text(fmt::format("bounces:\t{}", mLastCounters.TotalBounces));
    widget.text(fmt::format("validQ:\t\t{}", mLastCounters.ValidQueries));
    widget.text(fmt::format("grid cells:\t{}", mLastCounters.GridNonEmptyCells));
    widget.text(fmt::format("binned Q:\t{}", mLastCounters.GridBinnedQueries));
    widget.text(fmt::format("stores:\t\t{}", mLastCounters.PhotonStores));
    widget.indent(-16.0f);
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
    float queryRadius = (mDebugMode >= 4) ? (mPhotonRadius.x * 10.f) : mPhotonRadius.x;
    qVar["CB"]["gQueryRadius"] = queryRadius;
    qVar["gPhotonQueries"] = mpQueryBuffer;
    qVar["gQueryAABBs"] = mpQueryAABBBuffer;
    if (mpDebugCounters) qVar["gCounters"] = mpDebugCounters;

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
    // For a separate accumulation RT pass using a dummy SBT (geometryCount=1), use zero contribution index.
    instanceDesc.instanceContributionToHitGroupIndex = 0;
    // Allow any-hit invocation on procedural queries by not forcing opaque.
    instanceDesc.flags = RtGeometryInstanceFlags::None;
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
        desc.addShaderLibrary(kShaderResolvePS);
        desc.psEntry("psMain");
        mpResolveFullScreen = FullScreenPass::create(mpDevice, desc);
    }

    auto var = mpResolveFullScreen->getRootVar();
    var["CB"]["gFrameDim"] = mFrameDim;
    var["CB"]["gFrameIndex"] = mFrameCount;
    var["CB"]["gDebugMode"] = mDebugMode;
    var["CB"]["gAccumScale"] = 65536.0f;
    var["gQueryAccumulation"] = mpQueryAccumulator;
    var["gPhotonQueries"] = mpQueryBuffer;

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

    // Bind debug counters (cleared once per frame in execute())
    if (mpDebugCounters) var["gCounters"] = mpDebugCounters;

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

    // Allocate/resize photon hit buffer and bind outputs
    if (!mpPhotonHits || mpPhotonHits->getElementCount() < actualPhotons)
    {
        // Use reflection to allocate with correct stride for PhotonHit.
        mpPhotonHits = mpDevice->createStructuredBuffer(
            var["gPhotonHits"].getType(),
            /* elementCount */ actualPhotons,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal, nullptr, false);
        mpPhotonHits->setName("SPPM PhotonHits");
    }
    var["gPhotonHits"] = mpPhotonHits;
    var["CB"]["gPhotonHitCapacity"] = (uint32_t)mpPhotonHits->getElementCount();
    if (mpPhotonHitCounter)
        var["gPhotonHitCounter"] = mpPhotonHitCounter;

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

void SPPM::intersectPhotonsPass(RenderContext* pRenderContext, uint photonHitCount)
{
    FALCOR_PROFILE(pRenderContext, "IntersectPhotons");

    if (photonHitCount == 0 || !mpQueryTLAS || !mpQueryBuffer || !mpQueryAccumulator || !mpPhotonHits)
        return;

    if (!mIntersectPass.pProgram)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderIntersectPhotons);
        desc.setMaxPayloadSize(sizeof(uint32_t));
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(1);

        // Create a dummy SBT with geometryCount=1 for our custom TLAS.
        mIntersectPass.pBindingTable = RtBindingTable::create(1, 1, 1);
        auto& sbt = mIntersectPass.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen", mpScene->getTypeConformances()));
        sbt->setMiss(0, desc.addMiss("miss"));

        // Hit group for procedural queries: anyhit + intersection.
    auto hg = desc.addHitGroup("queryClosestHit", "queryAnyHit", "queryIntersection");
        sbt->setHitGroup(0, 0u, hg);

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        mIntersectPass.pProgram = Program::create(mpDevice, desc, defines);
    }

    if (!mIntersectPass.pVars)
        mIntersectPass.initProgramVars(mpDevice, mpScene, mpSampleGenerator);

    auto var = mIntersectPass.pVars->getRootVar();
    var["CB"]["gQueryCount"] = mQueryCount;
    var["CB"]["gAccumScale"] = 65536.0f;
    var["CB"]["gFirstPhoton"] = 0u;
    var["CB"]["gPhotonCount"] = photonHitCount;

    var["gQueryAS"].setAccelerationStructure(mpQueryTLAS);
    var["gPhotonQueries"] = mpQueryBuffer;
    var["gQueryAccumulation"] = mpQueryAccumulator;
    var["gPhotonHits"] = mpPhotonHits;
    if (mpDebugCounters) var["gCounters"] = mpDebugCounters;

    // Dispatch one ray per photon hit.
    mpScene->raytrace(pRenderContext, mIntersectPass.pProgram.get(), mIntersectPass.pVars, uint3(photonHitCount, 1, 1));
}

void SPPM::buildQueryGrid(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "BuildQueryGrid");
    if (!mpQueryBuffer || mQueryCount == 0) return;

    // Derive grid from scene bounds and query radius.
    const AABB& sceneBB = mpScene->getSceneBounds();
    float3 minP = sceneBB.minPoint;
    float3 maxP = sceneBB.maxPoint;
    float3 extent = maxP - minP;
    float radius = mPhotonRadius.x;
    // Ensure non-zero cell size.
    mCellSize = std::max(radius, 1e-3f);
    // Aim for ~64 cells along the longest axis, clamped.
    uint maxAxis = 0; float maxLen = extent.x;
    if (extent.y > maxLen) { maxLen = extent.y; maxAxis = 1; }
    if (extent.z > maxLen) { maxLen = extent.z; maxAxis = 2; }
    float targetCells = std::clamp(maxLen / mCellSize, 16.0f, 128.0f);
    mCellSize = std::max(maxLen / targetCells, radius);

    mGridMin = minP;
    uint3 dim;
    dim.x = std::max(1u, (uint)std::ceil(extent.x / mCellSize));
    dim.y = std::max(1u, (uint)std::ceil(extent.y / mCellSize));
    dim.z = std::max(1u, (uint)std::ceil(extent.z / mCellSize));
    // Clamp to avoid runaway memory.
    dim.x = std::min(dim.x, 512u);
    dim.y = std::min(dim.y, 512u);
    dim.z = std::min(dim.z, 512u);
    mGridDim = dim;

    const uint64_t cellCount = (uint64_t)mGridDim.x * mGridDim.y * mGridDim.z;

    // Allocate/resize buffers
    if (!mpGridHeads || mpGridHeads->getElementCount() != cellCount)
    {
        mpGridHeads = mpDevice->createStructuredBuffer(sizeof(int32_t), (uint32_t)cellCount,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, nullptr, false);
        mpGridHeads->setName("SPPM Grid Heads");
    }
    if (!mpGridNext || mpGridNext->getElementCount() != mQueryCount)
    {
        mpGridNext = mpDevice->createStructuredBuffer(sizeof(int32_t), mQueryCount,
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, nullptr, false);
        mpGridNext->setName("SPPM Grid Next");
    }

    // Clear grid heads to -1
    pRenderContext->clearUAV(mpGridHeads->getUAV().get(), uint4(0xFFFFFFFF));

    // Create and dispatch build pass
    if (!mpBuildGridPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderBuildQueryGrid).csEntry("main");
        DefineList defines; defines.add(mpScene->getSceneDefines());
        mpBuildGridPass = ComputePass::create(mpDevice, desc, defines);
    }

    auto var = mpBuildGridPass->getRootVar();
    var["CB"]["gGridDim"] = mGridDim;
    var["CB"]["gGridMin"] = mGridMin;
    var["CB"]["gCellSize"] = mCellSize;
    var["CB"]["gQueryCount"] = mQueryCount;
    var["gPhotonQueries"] = mpQueryBuffer;
    var["gGridHeads"] = mpGridHeads;
    var["gGridNext"] = mpGridNext;
    if (mpDebugCounters) var["gCounters"] = mpDebugCounters;

    uint32_t threads = mQueryCount;
    uint32_t groups = (threads + 255u) / 256u;
    mpBuildGridPass->execute(pRenderContext, uint3(groups, 1, 1));

    logInfo("SPPM Grid: dim=({}, {}, {}), cellSize={:.6f}", mGridDim.x, mGridDim.y, mGridDim.z, mCellSize);
}

void SPPM::accumulateByGrid(RenderContext* pRenderContext, uint photonHitCount)
{
    FALCOR_PROFILE(pRenderContext, "GridAccum");
    if (photonHitCount == 0 || !mpPhotonHits || !mpQueryBuffer || !mpQueryAccumulator || !mpGridHeads || !mpGridNext) return;

    if (!mpGridAccumPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderAccumulateByGrid).csEntry("main");
        DefineList defines; defines.add(mpScene->getSceneDefines());
        mpGridAccumPass = ComputePass::create(mpDevice, desc, defines);
    }

    auto var = mpGridAccumPass->getRootVar();
    var["CB"]["gPhotonCount"] = photonHitCount;
    var["CB"]["gAccumScale"] = 65536.0f;
    var["CB"]["gGridDim"] = mGridDim;
    var["CB"]["gGridMin"] = mGridMin;
    var["CB"]["gCellSize"] = mCellSize;

    var["gPhotonHits"] = mpPhotonHits;
    var["gPhotonQueries"] = mpQueryBuffer;
    var["gGridHeads"] = mpGridHeads;
    var["gGridNext"] = mpGridNext;
    var["gQueryAccumulation"] = mpQueryAccumulator;
    if (mpDebugCounters) var["gCounters"] = mpDebugCounters;

    uint32_t threads = photonHitCount;
    uint32_t groups = (threads + 255u) / 256u;
    mpGridAccumPass->execute(pRenderContext, uint3(groups, 1, 1));
}
