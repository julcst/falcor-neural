#include "TracePhotons.h"
#include "Structs.slang"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, TracePhotons>();
}

namespace
{
const char kShaderFile[] = "RenderPasses/TracePhotons/TracePhotons.rt.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 1u;

const std::string kPhotonBuffer = "photons";
const std::string kCounterBuffer = "counters";
} // namespace

TracePhotons::TracePhotons(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

Properties TracePhotons::getProperties() const
{
    return {};
}

RenderPassReflection TracePhotons::reflect(const CompileData& compileData)
{
    // Define the required resources here
    logInfo("TracePhotons::reflect called");
    RenderPassReflection reflector;
    reflector.addOutput(kPhotonBuffer, "Traced photons")
             .rawBuffer(mPhotonCount * mMaxBounces * sizeof(PhotonHit))
             .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addOutput(kCounterBuffer, "Photon Counters")
             .rawBuffer(sizeof(PhotonCounters))
             .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    return reflector;
}

void TracePhotons::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "TracePhotons");

    if (!mpScene) return;

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    prepareLightingStructure(pRenderContext);

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars)
        prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    // Set constants.
    const auto photonCapacity = mPhotonCount * mMaxBounces;
    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gMaxBounces"] = mMaxBounces;
    var["CB"]["gGlobalRejectionProb"] = mGlobalRejectionProb;
    var["CB"]["gPhotonHitCapacity"] = photonCapacity;

    if (mpEmissiveSampler)
        mpEmissiveSampler->bindShaderData(var["gLights"]["emissiveSampler"]);

    const auto& pPhotonHits = renderData.getResource(kPhotonBuffer)->asBuffer();
    FALCOR_ASSERT(pPhotonHits);
    var["gPhotonHits"] = pPhotonHits;

    const auto& pCounters = renderData.getResource(kCounterBuffer)->asBuffer();
    FALCOR_ASSERT(pCounters);
    pRenderContext->clearUAV(pCounters->getUAV().get(), uint4(0));
    var["gCounters"] = pCounters;
    
    // Trace photons
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(mPhotonCount, 1, 1));

#ifdef _DEBUG
    // pRenderContext->uavBarrier(pCounters.get());
    const auto counters = pCounters->getElement<PhotonCounters>(0u);
    logInfo("photons={} dispatched={} emitted={} stored={}", mPhotonCount, counters.PhotonsDispatched, counters.PhotonsEmitted, counters.PhotonStores);
#endif

    mFrameCount++;
}

void TracePhotons::renderUI(Gui::Widgets& widget) {}

void TracePhotons::prepareLightingStructure(RenderContext* pRenderContext)
{
    if (!mpScene) { return; }

    // Ensure emissive light collection is up to date on CPU side for correct flags/info.
    if (auto pLights = mpScene->getILightCollection(pRenderContext))
    {
        pLights->prepareSyncCPUData(pRenderContext);
    }

    const bool emissiveUsed = mpScene->useEmissiveLights();

    // Create/update emissive light sampler if emissive lights are present.
    if (emissiveUsed)
    {
        if (!mpEmissiveSampler)
        {
            auto pLights = mpScene->getILightCollection(pRenderContext);

            if (pLights)
            {
                mpEmissiveSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, pLights);
            }
        } else {
            mpEmissiveSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
        }
    }
    else
    {
        mpEmissiveSampler.reset();
    }
}

void TracePhotons::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;

    // Set new scene.
    mpScene = pScene;

    if (mpScene)
    {
        // Create ray tracing program.
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

void TracePhotons::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    if (mpEmissiveSampler)
    {
        mTracer.pProgram->addDefines(mpEmissiveSampler->getDefines());
    }
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}
