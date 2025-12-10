#include "AccumulatePhotonsRTX.h"
#include "../TracePhotons/Structs.slang"
#include "../TraceQueries/Query.slang"
#include "Structs.slang"
#include "Utils/StringUtils.h"

namespace
{
const char kQuerySearch[] = "RenderPasses/AccumulatePhotonsRTX/QuerySearch.rt.slang";
const char kPhotonSearch[] = "RenderPasses/AccumulatePhotonsRTX/PhotonSearch.rt.slang";
const char kPreparationComputeShaderFile[] = "RenderPasses/AccumulatePhotonsRTX/Preparation.cs.slang";
const char kPreparePhotonsComputeShaderFile[] = "RenderPasses/AccumulatePhotonsRTX/PreparePhotons.cs.slang";
const char kFinalizeTextureComputeShaderFile[] = "RenderPasses/AccumulatePhotonsRTX/FinalizeTexture.cs.slang";
const char kFinalizeBufferComputeShaderFile[] = "RenderPasses/AccumulatePhotonsRTX/FinalizeBuffer.cs.slang";


// Inputs
const char kQueryBuffer[] = "queries";
const char kPhotonBuffer[] = "photons";
const char kPhotonCounters[] = "photonCounters";

// Internal
const char kQueryAABBBuffer[] = "queryAABBs";
const char kPhotonAABBBuffer[] = "photonAABBs";
const char kQuerySphereBuffer[] = "querySpheres";
const char kQueryStateBuffer[] = "queryStates";
const char kAccumulatorBuffer[] = "accumulator";
const char kDebugCounters[] = "debugCounters";

// Output
const char kOutputTexture[] = "outputTexture";
const char kOutputBuffer[] = "outputBuffer";

// Properties
const char kVisualizeHeatmap[] = "visualizeHeatmap";
const char kReverseSearch[] = "reverseSearch";
const char kGlobalAlpha[] = "globalAlpha";
const char kCausticAlpha[] = "causticAlpha";
const char kGlobalRadius[] = "globalRadius";
const char kCausticRadius[] = "causticRadius";
const char kMaxNormalDeviation[] = "maxNormalDeviation";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxRecursionDepth = 1u;
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, AccumulatePhotonsRTX>();
}

AccumulatePhotonsRTX::AccumulatePhotonsRTX(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    setProperties(props);
}

void AccumulatePhotonsRTX::setProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kVisualizeHeatmap) mVisualizeHeatmap = value;
        else if (key == kReverseSearch) mReverseSearch = value;
        else if (key == kGlobalAlpha) mGlobalAlpha = value;
        else if (key == kCausticAlpha) mCausticAlpha = value;
        else if (key == kGlobalRadius) mGlobalRadius = value;
        else if (key == kCausticRadius) mCausticRadius = value;
        else if (key == kMaxNormalDeviation) mMaxNormalDeviation = value;
        else logWarning("Unrecognized property '{}' in AccumulatePhotonsRTX render pass.", key);
    }
}

Properties AccumulatePhotonsRTX::getProperties() const
{
    Properties props;
    props[kVisualizeHeatmap] = mVisualizeHeatmap;
    props[kReverseSearch] = mReverseSearch;
    props[kGlobalAlpha] = mGlobalAlpha;
    props[kCausticAlpha] = mCausticAlpha;
    props[kGlobalRadius] = mGlobalRadius;
    props[kCausticRadius] = mCausticRadius;
    props[kMaxNormalDeviation] = mMaxNormalDeviation;
    return props;
}

void AccumulatePhotonsRTX::renderUI(Gui::Widgets& widget) {
    widget.checkbox("Visualize Heatmap", mVisualizeHeatmap);
    if (widget.checkbox("Reverse Search", mReverseSearch)) requestRecompile();
    widget.var("Global Radius", mGlobalRadius, 0.0001f, 1.0f, 0.0001f);
    widget.var("Caustic Radius", mCausticRadius, 0.0001f, 1.0f, 0.0001f);
    widget.var("Global Alpha", mGlobalAlpha, 0.1f, 1.0f, 0.01f);
    widget.var("Caustic Alpha", mCausticAlpha, 0.1f, 1.0f, 0.01f);
    widget.var("Max Normal Deviation (degrees)", mMaxNormalDeviation, 0.0f, 90.0f, 1.0f);
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
        .rawBuffer(sizeof(PhotonCounters))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    reflector.addInternal(kDebugCounters, "Buffer for debug counters.")
        .rawBuffer(sizeof(DebugCounters))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInternal(kAccumulatorBuffer, "Buffer to accumulate outgoing flux and photon counts per query.")
        .rawBuffer(mQueryCount * sizeof(DoubleAccumulator))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addInternal(kQueryAABBBuffer, "Buffer containing ray query AABBs.")
        .rawBuffer(mQueryCount * sizeof(AABB))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    if (mReverseSearch) {
        reflector.addInternal(kPhotonAABBBuffer, "Buffer containing photon AABBs.")
            .rawBuffer(mPhotonHitCount * sizeof(AABB) / 4)
            .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    }
    reflector.addInternal(kQuerySphereBuffer, "Buffer for query geometry..")
        .rawBuffer(mQueryCount * sizeof(Sphere))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    reflector.addOutput(kQueryStateBuffer, "Buffer to keep radius and flux.")
        .rawBuffer(mQueryCount * sizeof(QueryState))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .flags(RenderPassReflection::Field::Flags::Persistent);
    
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

void AccumulatePhotonsRTX::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene)
        return;

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    auto shouldReset = is_set(mpScene->getUpdates(), IScene::UpdateFlags::AllButCamera); // Reset accumulation on scene change
    shouldReset |= mGlobalPhotonCounter == 0u; // Or in first frame
    
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
    
    const auto pPhotonAABBBuffer = renderData.getBuffer(kPhotonAABBBuffer);
    FALCOR_ASSERT(pPhotonAABBBuffer || !mReverseSearch);
    
    const auto counters = pPhotonCounters->getElement<PhotonCounters>(0u);

    if (mReverseSearch) {
        FALCOR_PROFILE(pRenderContext, "PreparePhotons");

        auto var = mpPreparePhotonsPass->getRootVar();
        var["gPhotonHits"] = pPhotonBuffer;
        var["gPhotonAABBs"] = pPhotonAABBBuffer;
        var["CB"]["gPhotonHitCount"] = counters.PhotonStores;
        var["CB"]["gRadius"] = std::max(mGlobalRadius, mCausticRadius);

        mpPreparePhotonsPass->execute(pRenderContext, counters.PhotonStores, 1, 1);
    }

    {
        FALCOR_PROFILE(pRenderContext, "PrepareQueries");

        auto var = mpPreparationPass->getRootVar();
        var["gQueries"] = pQueryBuffer;
        var["gQueryStates"] = pQueryStateBuffer;
        var["gQueryAABBs"] = pQueryAABBBuffer;
        var["gQuerySpheres"] = pQuerySphereBuffer;
        var["CB"]["gGlobalPhotonCount"] = mGlobalPhotonCounter;
        var["CB"]["gQueryCount"] = mQueryCount;
        var["CB"]["gReset"] = shouldReset;
        var["CB"]["gGlobalRadius"] = mGlobalRadius;
        var["CB"]["gCausticRadius"] = mCausticRadius;

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

    // Build query acceleration structure
    if (mReverseSearch)
        buildAccelerationStructure(pRenderContext, pPhotonAABBBuffer, counters.PhotonStores);
    else
        buildAccelerationStructure(pRenderContext, pQueryAABBBuffer, mQueryCount);

    FALCOR_ASSERT(mpTLAS);

    if (mReverseSearch) {
        FALCOR_PROFILE(pRenderContext, "PhotonSearch");
        
        if (!mTracer.pVars) prepareVars();
        auto var = mTracer.pVars->getRootVar();

        var["gPhotonAS"].setAccelerationStructure(mpTLAS);
        var["gQueries"] = pQueryBuffer;
        var["gQuerySpheres"] = pQuerySphereBuffer;
        var["gPhotonHits"] = pPhotonBuffer;
        var["gDebugCounters"] = pDebugCounters;
        var["gQueryAccumulation"] = pAccumulatorBuffer;
        var["CB"]["gQueryCount"] = mQueryCount;

        pRenderContext->clearUAV(pDebugCounters->getUAV().get(), uint4(0));
        
        logInfo("Tracing {} queries", mQueryCount);
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(mQueryCount, 1, 1));
    } else {
        FALCOR_PROFILE(pRenderContext, "QuerySearch");
        
        if (!mTracer.pVars) prepareVars();
        auto var = mTracer.pVars->getRootVar();

        var["gQueryAS"].setAccelerationStructure(mpTLAS);
        var["gPhotonQueries"] = pQueryBuffer;
        var["gQuerySpheres"] = pQuerySphereBuffer;
        var["gPhotonHits"] = pPhotonBuffer;
        var["gCounters"] = pPhotonCounters;
        var["gDebugCounters"] = pDebugCounters;
        var["gQueryAccumulation"] = pAccumulatorBuffer;
        var["CB"]["gMaxNormalDeviation"] = cosf(mMaxNormalDeviation * (M_PI_2 / 90.0f)); // Convert degrees to radians

        // Dispatch one ray per photon hit.
        logInfo("Tracing {} photons", counters.PhotonStores);
        pRenderContext->clearUAV(pAccumulatorBuffer->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(pDebugCounters->getUAV().get(), uint4(0));
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(counters.PhotonStores, 1, 1));
    }

    pRenderContext->uavBarrier(pDebugCounters.get());
    const auto debugCounters = pDebugCounters->getElement<DebugCounters>(0u);
    logInfo("IntersectorCalls={}, Accumulations={}",
        debugCounters.IntersectorCalls,
        debugCounters.Accumulations);

    {
        FALCOR_PROFILE(pRenderContext, "ComputeFinalRadiance");

        const auto pOutputTexture = renderData.getTexture(kOutputTexture);
        const auto pOutputBuffer = renderData[kOutputBuffer];

        mGlobalPhotonCounter += counters.PhotonsEmitted;
        renderData.getDictionary()["GlobalPhotonCount"] = mGlobalPhotonCounter; // For other passes to read

        if (pOutputTexture)
        {
            auto var = mpFinalizeTexturePass->getRootVar();
            var["gAccumulator"] = pAccumulatorBuffer;
            var["gPhotonQueries"] = pQueryBuffer;
            var["gQueryStates"] = pQueryStateBuffer;
            var["gOutputTexture"] = pOutputTexture;
            var["CB"]["gGlobalPhotonCount"] = mGlobalPhotonCounter;
            var["CB"]["gFrameDim"] = renderData.getDefaultTextureDims();
            var["CB"]["gVisualizeHeatmap"] = mVisualizeHeatmap;
            var["CB"]["gGlobalAlpha"] = mGlobalAlpha;
            var["CB"]["gCausticAlpha"] = mCausticAlpha;

            logInfo("Collecting flux (Texture), globalPhotons={}", mGlobalPhotonCounter);
            mpFinalizeTexturePass->execute(pRenderContext, uint3(renderData.getDefaultTextureDims(), 1));
        }
        else if (pOutputBuffer)
        {
            auto var = mpFinalizeBufferPass->getRootVar();
            var["gAccumulator"] = pAccumulatorBuffer;
            var["gPhotonQueries"] = pQueryBuffer;
            var["gQueryStates"] = pQueryStateBuffer;
            var["gOutputBuffer"] = pOutputBuffer->asBuffer();
            var["CB"]["gGlobalPhotonCount"] = mGlobalPhotonCounter;
            var["CB"]["gQueryCount"] = mQueryCount;
            var["CB"]["gGlobalAlpha"] = mGlobalAlpha;
            var["CB"]["gCausticAlpha"] = mCausticAlpha;

            logInfo("Collecting flux (Buffer), globalPhotons={}", mGlobalPhotonCounter);
            mpFinalizeBufferPass->execute(pRenderContext, mQueryCount, 1, 1);
        }
    }
}

void AccumulatePhotonsRTX::buildAccelerationStructure(RenderContext* pRenderContext, const ref<Buffer>& pAABBBuffer, const uint32_t aabbCount)
{
    FALCOR_PROFILE(pRenderContext, "BuildQueryAS");
    FALCOR_ASSERT(pAABBBuffer);
    static_assert(sizeof(AABB) == sizeof(RtAABB));

    // AABBs already written by TraceQueries.

    RtGeometryDesc geometryDesc = {
        .type = RtGeometryType::ProcedurePrimitives,
        .flags = RtGeometryFlags::None,
        .content = {
            .proceduralAABBs = {
                .count = aabbCount,
                .data = pAABBBuffer->getGpuAddress(),
                .stride = sizeof(RtAABB),
            }
        }
    };

    // TODO: Optimize size
    RtAccelerationStructureBuildInputs blasInputs = {
        .kind = RtAccelerationStructureKind::BottomLevel,
        .flags = RtAccelerationStructureBuildFlags::PreferFastBuild | RtAccelerationStructureBuildFlags::MinimizeMemory,
        .descCount = 1,
        .geometryDescs = &geometryDesc,
    };

    const auto blasPrebuild = RtAccelerationStructure::getPrebuildInfo(mpDevice.get(), blasInputs);

    auto ensureASBuffer = [&](ref<Buffer>& buffer, uint64_t requiredSize, ResourceBindFlags flags, const char* name)
    {
        if (!buffer || buffer->getSize() < requiredSize)
        {
            buffer = mpDevice->createBuffer(requiredSize, flags, MemoryType::DeviceLocal);
            buffer->setName(name);
        }
    };

    logInfo("BLAS size={}, scratch size={}", formatByteSize(blasPrebuild.resultDataMaxSize), formatByteSize(blasPrebuild.scratchDataSize));
    ensureASBuffer(mpBlasStorage, blasPrebuild.resultDataMaxSize, ResourceBindFlags::AccelerationStructure, "SPPM Query BLAS");
    ensureASBuffer(mpBlasScratch, blasPrebuild.scratchDataSize, ResourceBindFlags::UnorderedAccess, "SPPM Query BLAS Scratch");
    
    RtAccelerationStructure::Desc blasDesc;
    blasDesc.setKind(RtAccelerationStructureKind::BottomLevel).setBuffer(mpBlasStorage, 0, mpBlasStorage->getSize());
    mpBLAS = RtAccelerationStructure::create(mpDevice, blasDesc);

    // TODO: Reuse BLAS when possible
    RtAccelerationStructure::BuildDesc blasBuild = {
        .inputs = blasInputs,
        .source = nullptr,
        .dest = mpBLAS.get(),
        .scratchData = mpBlasScratch->getGpuAddress(),
    };

    pRenderContext->buildAccelerationStructure(blasBuild, 0, nullptr);
    pRenderContext->submit();
    pRenderContext->uavBarrier(mpBlasStorage.get());

    RtInstanceDesc instanceDesc = {
        .transform = { 1.f, 0.f, 0.f, 0.f,
                       0.f, 1.f, 0.f, 0.f,
                       0.f, 0.f, 1.f, 0.f },
        .instanceID = 0,
        .instanceMask = 0xFF,
        .instanceContributionToHitGroupIndex = 0,
        .flags = RtGeometryInstanceFlags::None,
        .accelerationStructure = mpBLAS->getGpuAddress(),
    };

    auto allocation = mpDevice->getUploadHeap()->allocate(sizeof(RtInstanceDesc), sizeof(RtInstanceDesc));
    std::memcpy(allocation.pData, &instanceDesc, sizeof(RtInstanceDesc));

    RtAccelerationStructureBuildInputs tlasInputs = {
        .kind = RtAccelerationStructureKind::TopLevel,
        .flags = RtAccelerationStructureBuildFlags::PreferFastTrace,
        .descCount = 1,
        .instanceDescs = allocation.getGpuAddress(),
    };

    const auto tlasPrebuild = RtAccelerationStructure::getPrebuildInfo(mpDevice.get(), tlasInputs);

    ensureASBuffer(mpTlasStorage, tlasPrebuild.resultDataMaxSize, ResourceBindFlags::AccelerationStructure, "SPPM Query TLAS");
    ensureASBuffer(mpTlasScratch, tlasPrebuild.scratchDataSize, ResourceBindFlags::UnorderedAccess, "SPPM Query TLAS Scratch");

    RtAccelerationStructure::Desc tlasDesc;
    tlasDesc.setKind(RtAccelerationStructureKind::TopLevel).setBuffer(mpTlasStorage, 0, mpTlasStorage->getSize());
    mpTLAS = RtAccelerationStructure::create(mpDevice, tlasDesc);

    RtAccelerationStructure::BuildDesc tlasBuild = {
        .inputs = tlasInputs,
        .source = nullptr,
        .dest = mpTLAS.get(),
        .scratchData = mpTlasScratch->getGpuAddress(),
    };

    pRenderContext->buildAccelerationStructure(tlasBuild, 0, nullptr);
    
    mpDevice->getUploadHeap()->release(allocation);
    
    pRenderContext->uavBarrier(mpTlasStorage.get());
}

void AccumulatePhotonsRTX::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mTracer.pProgram.reset();
    mTracer.pBindingTable.reset();
    mTracer.pVars.reset();
    mpScene = pScene;

    if (mpScene)
    {
        if (mReverseSearch) {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kPhotonSearch);
            desc.setMaxPayloadSize(sizeof(PhotonSearchPayload));
            desc.setMaxAttributeSize(sizeof(PhotonIsectAttribs));
            desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

            // Create a dummy SBT with geometryCount=1 for our custom TLAS.
            mTracer.pBindingTable = RtBindingTable::create(0, 1, 1);
            auto& sbt = mTracer.pBindingTable;
            sbt->setRayGen(desc.addRayGen("rayGen", mpScene->getTypeConformances()));

            // Hit group for procedural queries: anyhit + intersection.
            sbt->setHitGroup(0, 0u, desc.addHitGroup("", "photonAnyHit", "photonIntersection", mpScene->getTypeConformances())); // single entry for our custom TLAS

            mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        } else {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kQuerySearch);
            desc.setMaxPayloadSize(sizeof(QuerySearchPayload));
            desc.setMaxAttributeSize(sizeof(QueryIsectAttribs));
            desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

            // Create a dummy SBT with geometryCount=1 for our custom TLAS.
            mTracer.pBindingTable = RtBindingTable::create(0, 1, 1);
            auto& sbt = mTracer.pBindingTable;
            sbt->setRayGen(desc.addRayGen("rayGen", mpScene->getTypeConformances()));

            // Hit group for procedural queries: anyhit + intersection.
            sbt->setHitGroup(0, 0u, desc.addHitGroup("", "queryAnyHit", "queryIntersection", mpScene->getTypeConformances())); // single entry for our custom TLAS

            mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
        }
    }

    if (!mpPreparationPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kPreparationComputeShaderFile);
        desc.csEntry("main");

        mpPreparationPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines()); // NOTE: Needs scene defines for hit types
    }

    if (!mpPreparePhotonsPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kPreparePhotonsComputeShaderFile);
        desc.csEntry("main");

        mpPreparePhotonsPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    }

    if (!mpFinalizeTexturePass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kFinalizeTextureComputeShaderFile);
        desc.csEntry("main");

        mpFinalizeTexturePass = ComputePass::create(mpDevice, desc);
    }

    if (!mpFinalizeBufferPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kFinalizeBufferComputeShaderFile);
        desc.csEntry("main");

        mpFinalizeBufferPass = ComputePass::create(mpDevice, desc);
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
