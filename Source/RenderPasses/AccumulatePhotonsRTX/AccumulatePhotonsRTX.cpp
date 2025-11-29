#include "AccumulatePhotonsRTX.h"
#include "../TracePhotons/Structs.slang"
#include "../TraceQueries/Query.slang"
#include "DebugCounters.slang"
#include "Structs.slang"

namespace
{
const char kShaderFile[] = "RenderPasses/AccumulatePhotonsRTX/AccumulatePhotons.rt.slang";
const char kPreparationComputeShaderFile[] = "RenderPasses/AccumulatePhotonsRTX/Preparation.cs.slang";
const char kFinalizeComputeShaderFile[] = "RenderPasses/AccumulatePhotonsRTX/Finalize.cs.slang";


// Inputs
const char kQueryBuffer[] = "queries";
const char kPhotonBuffer[] = "photons";
const char kPhotonCounters[] = "photonCounters";

// Internal
const char kQueryAABBBuffer[] = "queryAABBs";
const char kQuerySphereBuffer[] = "querySpheres";
const char kQueryStateBuffer[] = "queryStates";
const char kAccumulatorBuffer[] = "accumulator";
const char kDebugCounters[] = "debugCounters";

// Output
const char kOutputTexture[] = "outputTexture";
const char kOutputBuffer[] = "outputBuffer";

// Properties
const char kVisualizeHeatmap[] = "visualizeHeatmap";
const char kQueryRadius[] = "radius";
const char kRadiusAlpha[] = "alpha";

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
        if (key == kVisualizeHeatmap) {
            mVisualizeHeatmap = value;
        } else if (key == kQueryRadius) {
            mQueryRadius = props[kQueryRadius];
        } else if (key == kRadiusAlpha) {
            mAlpha = props[kRadiusAlpha];
        } else {
            logWarning("Unrecognized property '{}' in AccumulatePhotonsRTX render pass.", key);
        }
    }
}

Properties AccumulatePhotonsRTX::getProperties() const
{
    Properties props;
    props[kVisualizeHeatmap] = mVisualizeHeatmap;
    props[kQueryRadius] = mQueryRadius;
    props[kRadiusAlpha] = mAlpha;
    return props;
}

RenderPassReflection AccumulatePhotonsRTX::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addInput(kQueryBuffer, "Buffer containing ray query results.")
        .rawBuffer(mQueryCount * sizeof(Query))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInput(kPhotonBuffer, "Buffer containing photons to be accumulated.")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInput(kPhotonCounters, "Buffer containing photon counters.")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    reflector.addInternal(kDebugCounters, "Buffer for debug counters.")
        .rawBuffer(sizeof(DebugCounters))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInternal(kAccumulatorBuffer, "Buffer to accumulate outgoing flux and photon counts per query.")
        .rawBuffer(mQueryCount * sizeof(float4))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInternal(kQueryAABBBuffer, "Buffer containing ray query AABBs.")
        .rawBuffer(mQueryCount * sizeof(AABB))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInternal(kQuerySphereBuffer, "Buffer for query geometry..")
        .rawBuffer(mQueryCount * sizeof(Sphere))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInternal(kQueryStateBuffer, "Buffer to keep radius and flux.")
        .rawBuffer(mQueryCount * sizeof(QueryState))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    
    reflector.addOutput(kOutputTexture, "Output texture showing accumulated radiance.")
        .texture2D(0, 0)
        .format(ResourceFormat::RGBA32Float)
        .flags(RenderPassReflection::Field::Flags::Optional);

    reflector.addOutput(kOutputBuffer, "Output radiance buffer")
        .rawBuffer(mQueryCount * sizeof(float3))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .flags(RenderPassReflection::Field::Flags::Optional);
    
    return reflector;
}

void AccumulatePhotonsRTX::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    for (auto i = 0; i < compileData.connectedResources.getFieldCount(); ++i) {
        const auto& field = compileData.connectedResources.getField(i);
        logInfo("Connected resource: {} (type={})", field->getName(), field->getWidth());
    }
    const auto queryCount = compileData.connectedResources.getField(kQueryBuffer)->getWidth() / sizeof(Query);
    logInfo("queryCount={}", queryCount);
    if (mQueryCount != queryCount) {
        mQueryCount = queryCount;
        FALCOR_CHECK(false, "Recompute query count"); // Force retry of reflect
    }
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
    const auto pQueryBuffer = renderData[kQueryBuffer]->asBuffer();
    const auto pQuerySphereBuffer = renderData[kQuerySphereBuffer]->asBuffer();
    const auto pQueryStateBuffer = renderData[kQueryStateBuffer]->asBuffer();
    const auto pPhotonBuffer = renderData[kPhotonBuffer]->asBuffer();
    const auto pPhotonCounters = renderData[kPhotonCounters]->asBuffer();
    const auto pAccumulatorBuffer = renderData[kAccumulatorBuffer]->asBuffer();
    const auto pDebugCounters = renderData[kDebugCounters]->asBuffer();
    FALCOR_ASSERT(pQueryAABBBuffer);
    FALCOR_ASSERT(pQuerySphereBuffer); FALCOR_ASSERT(pQueryStateBuffer);
    FALCOR_ASSERT(pQueryBuffer); FALCOR_ASSERT(pAccumulatorBuffer);
    FALCOR_ASSERT(pPhotonBuffer); FALCOR_ASSERT(pPhotonCounters);
    FALCOR_ASSERT(pDebugCounters);
    FALCOR_ASSERT_GE(pAccumulatorBuffer->getSize(), mQueryCount * sizeof(float4));

    {
        FALCOR_PROFILE(pRenderContext, "PrepareQueries");

        auto var = mpPreparationPass->getRootVar();
        var["gQueries"] = pQueryBuffer;
        var["gQueryStates"] = pQueryStateBuffer;
        var["gQueryAABBs"] = pQueryAABBBuffer;
        var["gQuerySpheres"] = pQuerySphereBuffer;
        var["CB"]["gQueryCount"] = mQueryCount;
        var["CB"]["gReset"] = mpScene->getUpdates() != IScene::UpdateFlags::None;
        var["CB"]["gInitialRadius"] = mQueryRadius;

        logInfo("Preparing {} queries", mQueryCount);
        mpPreparationPass->execute(pRenderContext, mQueryCount, 1);
    }

    pRenderContext->uavBarrier(pQueryAABBBuffer.get()); // TODO: Is this necessary?
    // const auto aabbs = pQueryAABBBuffer->getElements<AABB>();
    // auto valid = 0u;
    // for (const auto aabb : aabbs) {
    //     if (aabb.valid()) valid += 1u;
    // }
    // logInfo("valid={}%", (100u * valid) / aabbs.size());

    {
        // Build query acceleration structure
        buildQueryAcceleration(pRenderContext, pQueryAABBBuffer);
        FALCOR_ASSERT(mpQueryTLAS);
    }

    {
        FALCOR_PROFILE(pRenderContext, "AccumulatePhotonsRTX");
        
        if (!mTracer.pVars) prepareVars();
        auto var = mTracer.pVars->getRootVar();

        var["gQueryAS"].setAccelerationStructure(mpQueryTLAS);
        var["gPhotonQueries"] = pQueryBuffer;
        var["gQuerySpheres"] = pQuerySphereBuffer;
        var["gPhotonHits"] = pPhotonBuffer;
        var["gCounters"] = pPhotonCounters;
        var["gDebugCounters"] = pDebugCounters;
        var["gQueryAccumulation"] = pAccumulatorBuffer;

        // TODO: Use indirect dispatch to avoid CPU-GPU sync
        pRenderContext->uavBarrier(pPhotonCounters.get());
        const auto counters = pPhotonCounters->getElement<PhotonCounters>(0u);
        const auto photonHitCount = counters.PhotonStores;
        mGlobalPhotonCounter += counters.PhotonsEmitted;

        // Dispatch one ray per photon hit.
        logInfo("Tracing {} photons", photonHitCount);
        pRenderContext->clearUAV(pAccumulatorBuffer->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(pDebugCounters->getUAV().get(), uint4(0));
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

        // Specialize program
        // These defines should not modify the program vars. Do not trigger program vars re-creation.
        const auto pOutputTexture = renderData.getTexture(kOutputTexture);
        mpFinalizePass->addDefine("OUTPUT_TEXTURE", pOutputTexture ? "1" : "0");
        const auto pOutputBuffer = renderData[kOutputBuffer];
        mpFinalizePass->addDefine("OUTPUT_BUFFER", pOutputBuffer ? "1" : "0");

        // All defines should be set up to this point
        auto var = mpFinalizePass->getRootVar();
        var["gAccumulator"] = pAccumulatorBuffer;
        var["gPhotonQueries"] = pQueryBuffer;
        var["gQueryStates"] = pQueryStateBuffer;
        if (pOutputTexture) var["gOutputTexture"] = pOutputTexture;
        if (pOutputBuffer) var["gOutputBuffer"] = pOutputBuffer;
        var["CB"]["gGlobalPhotonCount"] = mGlobalPhotonCounter;
        var["CB"]["gFrameDim"] = renderData.getDefaultTextureDims();
        var["CB"]["gVisualizeHeatmap"] = mVisualizeHeatmap;
        var["CB"]["gAlpha"] = mAlpha;

        //pRenderContext->uavBarrier(pAccumulatorBuffer.get());
        mpFinalizePass->execute(pRenderContext, uint3(renderData.getDefaultTextureDims(), 1));
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

    if (!mpPreparationPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kPreparationComputeShaderFile);
        desc.csEntry("main");

        mpPreparationPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines()); // NOTE: Needs scene defines for hit types
    }

    if (!mpFinalizePass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kFinalizeComputeShaderFile);
        desc.csEntry("main");

        mpFinalizePass = ComputePass::create(mpDevice, desc);
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
