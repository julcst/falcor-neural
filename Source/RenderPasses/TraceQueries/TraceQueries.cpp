#include "TraceQueries.h"
#include "Query.slang"

namespace
{
const char kShaderFile[] = "RenderPasses/TraceQueries/TraceQueries.rt.slang";
const char kQueryRadius[] = "queryRadius";
const char kQueries[] = "queries";
const char kAABBs[] = "aabbs";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = sizeof(uint4);
const uint32_t kMaxRecursionDepth = 1u;
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, TraceQueries>();
}

TraceQueries::TraceQueries(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
    setProperties(props);
}

Properties TraceQueries::getProperties() const
{
    Properties props;
    props[kQueryRadius] = mQueryRadius;
    return props;
}

void TraceQueries::setProperties(const Properties& props)
{
    if (props.has(kQueryRadius))
        mQueryRadius = props[kQueryRadius];
}

// This should recompile on resolution change, adjusting buffer sizes
RenderPassReflection TraceQueries::reflect(const CompileData& compileData)
{
    const uint2 dims = compileData.defaultTexDims;
    uint32_t queryCount = dims.x * dims.y;

    RenderPassReflection reflector;

    reflector.addOutput(kQueries, "Per-pixel photon queries")
        .rawBuffer(queryCount * sizeof(Query))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    reflector.addOutput(kAABBs, "Per-pixel query AABBs")
        .rawBuffer(queryCount * sizeof(AABB))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    return reflector;
}

void TraceQueries::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "TraceQueries");

    if (!mpScene)
        return;

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    if (!mTracer.pVars)
        prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    const uint2 frameDim = renderData.getDefaultTextureDims();
    if (frameDim.x == 0 || frameDim.y == 0)
        return;
    const uint32_t queryCount = frameDim.x * frameDim.y;
    if (queryCount == 0)
        return;

    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameDim"] = frameDim;
    var["CB"]["gFrameIndex"] = mFrameCount;
    var["CB"]["gQueryRadius"] = mQueryRadius;

    // Get the buffers allocated by the RenderGraph reflection.
    const auto& pQueryResource = renderData.getResource(kQueries);
    const auto& pQueryAABBResource = renderData.getResource(kAABBs);
    FALCOR_ASSERT(pQueryResource);
    FALCOR_ASSERT(pQueryAABBResource);
    auto pQueryBuffer = pQueryResource->asBuffer();
    auto pQueryAABBBuffer = pQueryAABBResource->asBuffer();
    FALCOR_ASSERT(pQueryBuffer);
    FALCOR_ASSERT(pQueryAABBBuffer);
    // TODO: Assert correct sizes

    var["gPhotonQueries"] = pQueryBuffer;
    var["gQueryAABBs"] = pQueryAABBBuffer;

    logInfo("sizeof(AABB)={} sizeof(Query)={}", var["gQueryAABBs"].getType()->getByteSize(), var["gPhotonQueries"].getType()->getByteSize());

    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(frameDim.x, frameDim.y, 1));

    mFrameCount++;
}

void TraceQueries::renderUI(Gui::Widgets& widget)
{
    widget.var("Query radius", mQueryRadius, 0.f, 100.f);
}

void TraceQueries::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;
    mpScene = pScene;

    if (mpScene)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.pBindingTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("miss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));
        }

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
    }
}

void TraceQueries::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program with generator defines and scene-specific type conformances.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program. This may trigger shader compilation.
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}
