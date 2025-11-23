#include "AccumulatePhotonsRTX.h"
#include "../TracePhotons/Structs.slang"
#include "DebugCounters.slang"

namespace
{
const char kShaderFile[] = "RenderPasses/AccumulatePhotonsRTX/AccumulatePhotons.rt.slang";
const char kComputeFile[] = "RenderPasses/AccumulatePhotonsRTX/Visualize.cs.slang";
const char kQueryBuffer[] = "queries";
const char kQueryAABBBuffer[] = "queryAABBs";
const char kPhotonBuffer[] = "photons";
const char kPhotonCounters[] = "photonCounters";
const char kAccumulatorBuffer[] = "accumulator";
const char kOutput[] = "output";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = sizeof(float3);
const uint32_t kMaxRecursionDepth = 1u;
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, AccumulatePhotonsRTX>();
}

AccumulatePhotonsRTX::AccumulatePhotonsRTX(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    for (const auto& [key, value] : props)
    {
        if (key == "visualizeHeatmap")
        {
            mVisualizeHeatmap = value;
        }
        else {
            logWarning("Unrecognized property '{}' in AccumulatePhotonsRTX render pass.", key);
        }
    }
}

Properties AccumulatePhotonsRTX::getProperties() const
{
    Properties props;
    props["visualizeHeatmap"] = mVisualizeHeatmap;
    return props;
}

RenderPassReflection AccumulatePhotonsRTX::reflect(const CompileData& compileData)
{
    const auto queryCount = 1166400; // TODO: Extract from dictionary
    logInfo("queryCount={}", queryCount);

    // Define the required resources here
    RenderPassReflection reflector;
    const auto& queries = reflector.addInput(kQueryBuffer, "Buffer containing ray query results.")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInput(kQueryAABBBuffer, "Buffer containing ray query AABBs.")
        .rawBuffer(queryCount * sizeof(AABB))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInput(kPhotonBuffer, "Buffer containing photons to be accumulated.")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInput(kPhotonCounters, "Buffer containing photon counters.")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    reflector.addInternal("debugCounters", "Buffer for debug counters.")
        .rawBuffer(sizeof(DebugCounters))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInternal(kAccumulatorBuffer, "Buffer to accumulate outgoing flux and photon counts per query.")
        .rawBuffer(queryCount * sizeof(float4)) // TODO: match query count
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    
    reflector.addOutput(kOutput, "Output texture showing accumulated radiance.")
        .texture2D(0, 0)
        .format(ResourceFormat::RGBA32Float);
    
    return reflector;
}

void AccumulatePhotonsRTX::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
        return;

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }
    
    // Get inputs
    const auto pQueryAABBBuffer = renderData[kQueryAABBBuffer]->asBuffer();
    const auto queryCount = pQueryAABBBuffer->getSize() / sizeof(AABB);
    FALCOR_ASSERT(pQueryAABBBuffer);
    logInfo("queryCount={}", queryCount);

    const auto aabbs = pQueryAABBBuffer->getElements<AABB>();
    auto valid = 0u;
    for (const auto aabb : aabbs) {
        if (aabb.valid()) valid += 1u;
    }
    logInfo("valid={}%", (100u * valid) / aabbs.size());

    // Build query acceleration structure
    buildQueryAcceleration(pRenderContext, pQueryAABBBuffer);

    const auto pQueryBuffer = renderData[kQueryBuffer]->asBuffer();
    const auto pPhotonBuffer = renderData[kPhotonBuffer]->asBuffer();
    const auto pPhotonCounters = renderData[kPhotonCounters]->asBuffer();
    const auto pAccumulatorBuffer = renderData[kAccumulatorBuffer]->asBuffer();
    const auto pDebugCounters = renderData["debugCounters"]->asBuffer();
    // FALCOR_ASSERT(mpQueryTLAS);
    FALCOR_ASSERT(pQueryBuffer); FALCOR_ASSERT(pAccumulatorBuffer);
    FALCOR_ASSERT(pPhotonBuffer); FALCOR_ASSERT(pPhotonCounters);
    FALCOR_ASSERT(pDebugCounters);

    // Intersect photons
    {
        FALCOR_PROFILE(pRenderContext, "IntersectPhotons");
        
        if (!mTracer.pVars) prepareVars();
        auto var = mTracer.pVars->getRootVar();

        var["gQueryAS"].setAccelerationStructure(mpQueryTLAS);
        var["gPhotonQueries"] = pQueryBuffer;
        var["gQueryAccumulation"] = pAccumulatorBuffer;
        var["gPhotonHits"] = pPhotonBuffer;
        var["gCounters"] = pPhotonCounters;
        var["gDebugCounters"] = pDebugCounters;

        // TODO: Use indirect dispatch to avoid CPU-GPU sync
        pRenderContext->uavBarrier(pPhotonCounters.get());
        const auto counters = pPhotonCounters->getElement<PhotonCounters>(0u);
        const auto photonHitCount = counters.PhotonStores;

        // Dispatch one ray per photon hit.
        logInfo("Tracing {} photons", photonHitCount);
        pRenderContext->clearUAV(pAccumulatorBuffer->getUAV().get(), uint4(0));
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(photonHitCount, 1, 1));
        
        pRenderContext->uavBarrier(pDebugCounters.get());
        const auto debugCounters = pDebugCounters->getElement<DebugCounters>(0u);
        logInfo("IntersectorCalls={}, RaygenCalls={}, Accumulations={}",
            debugCounters.IntersectorCalls,
            debugCounters.RaygenCalls,
            debugCounters.Accumulations);
    }

    {
        FALCOR_PROFILE(pRenderContext, "ComputeFinalRadiance");

        auto var = mpVisualizePass->getRootVar();
        var["gAccumulator"] = pAccumulatorBuffer;
        var["gOutput"] = renderData.getTexture(kOutput);
        var["gPhotonQueries"] = pQueryBuffer;
        var["gCounters"] = pPhotonCounters;
        var["CB"]["gFrameDim"] = renderData.getDefaultTextureDims();
        var["CB"]["gVisualizeHeatmap"] = mVisualizeHeatmap;

        //pRenderContext->uavBarrier(pAccumulatorBuffer.get());
        mpVisualizePass->execute(pRenderContext, uint3(renderData.getDefaultTextureDims(), 1));
    }
}

void AccumulatePhotonsRTX::buildQueryAcceleration(RenderContext* pRenderContext, ref<Buffer> pQueryAABBBuffer)
{
    FALCOR_PROFILE(pRenderContext, "BuildQueryAS");
    FALCOR_ASSERT(pQueryAABBBuffer);
    static_assert(sizeof(AABB) == sizeof(RtAABB));

    // AABBs already written by TraceQueries.

    RtGeometryDesc geometryDesc = {};
    geometryDesc.type = RtGeometryType::ProcedurePrimitives;
    geometryDesc.flags = RtGeometryFlags::None;
    geometryDesc.content.proceduralAABBs.count = pQueryAABBBuffer->getSize() / sizeof(RtAABB);
    geometryDesc.content.proceduralAABBs.data = pQueryAABBBuffer->getGpuAddress();
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
            buffer = mpDevice->createBuffer(requiredSize, flags, MemoryType::DeviceLocal);
            buffer->setName(name);
        }
    };

    logInfo("BLAS size={}, scratch size={}", blasPrebuild.resultDataMaxSize, blasPrebuild.scratchDataSize);
    ensureASBuffer(mpQueryBlasStorage, blasPrebuild.resultDataMaxSize, ResourceBindFlags::AccelerationStructure, "SPPM Query BLAS");
    ensureASBuffer(mpQueryBlasScratch, blasPrebuild.scratchDataSize, ResourceBindFlags::UnorderedAccess, "SPPM Query BLAS Scratch");
    
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
    pRenderContext->submit();
    pRenderContext->uavBarrier(mpQueryBlasStorage.get());

    RtInstanceDesc instanceDesc = {
        .transform = { 1.f, 0.f, 0.f, 0.f,
                       0.f, 1.f, 0.f, 0.f,
                       0.f, 0.f, 1.f, 0.f },
        .instanceID = 0,
        .instanceMask = 0xFF,
        .instanceContributionToHitGroupIndex = 0,
        .flags = RtGeometryInstanceFlags::None,
        .accelerationStructure = mpQueryBLAS->getGpuAddress(),
    };

    RtAccelerationStructureBuildInputs tlasInputs = {};
    tlasInputs.kind = RtAccelerationStructureKind::TopLevel;
    tlasInputs.flags = RtAccelerationStructureBuildFlags::PreferFastTrace;
    tlasInputs.descCount = 1;

    auto allocation = mpDevice->getUploadHeap()->allocate(sizeof(RtInstanceDesc), sizeof(RtInstanceDesc));
    std::memcpy(allocation.pData, &instanceDesc, sizeof(RtInstanceDesc));
    tlasInputs.instanceDescs = allocation.getGpuAddress();

    const auto tlasPrebuild = RtAccelerationStructure::getPrebuildInfo(mpDevice.get(), tlasInputs);

    ensureASBuffer(mpQueryTlasStorage, tlasPrebuild.resultDataMaxSize, ResourceBindFlags::AccelerationStructure, "SPPM Query TLAS");
    ensureASBuffer(mpQueryTlasScratch, tlasPrebuild.scratchDataSize, ResourceBindFlags::UnorderedAccess, "SPPM Query TLAS Scratch");

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
    
    mpDevice->getUploadHeap()->release(allocation);
    
    pRenderContext->uavBarrier(mpQueryTlasStorage.get());
}

void AccumulatePhotonsRTX::renderUI(Gui::Widgets& widget) {}

void AccumulatePhotonsRTX::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mTracer.pProgram.reset();
    mTracer.pBindingTable.reset();
    mTracer.pVars.reset();
    mpScene = pScene;

    if (mpScene)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(sizeof(float));
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        // Create a dummy SBT with geometryCount=1 for our custom TLAS.
        mTracer.pBindingTable = RtBindingTable::create(0, 1, 1);
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen", mpScene->getTypeConformances()));

        // Hit group for procedural queries: anyhit + intersection.
        sbt->setHitGroup(0, 0u, desc.addHitGroup("", "queryAnyHit", "queryIntersection", mpScene->getTypeConformances())); // single entry for our custom TLAS

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
    }

    if (!mpVisualizePass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kComputeFile);
        desc.csEntry("main");

        mpVisualizePass = ComputePass::create(mpDevice, desc);
    }
}

void AccumulatePhotonsRTX::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program with generator defines and scene-specific type conformances.
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program. This may trigger shader compilation.
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
}
