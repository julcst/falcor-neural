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
const uint32_t kMaxPayloadSizeBytes = 3u * sizeof(float4);
const uint32_t kMaxRecursionDepth = 1u;

const std::string kPhotonBuffer = "photons";
const std::string kCounterBuffer = "counters";

// Propertiesvalue
const std::string kPhotonCount = "photonCount";
const std::string kMaxBounces = "maxBounces";
const std::string kGlobalRejectionProb = "globalRejectionProb";
const std::string kRussianRouletteWeight = "russianRouletteWeight";
const std::string kUseRussianRoulette = "useRussianRoulette";
const std::string kUseWaveIntrinsics = "useWaveIntrinsics";
} // namespace

TracePhotons::TracePhotons(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
    setProperties(props);
}

void TracePhotons::setProperties(const Properties& props)
{
    for (const auto& [key, value] : props) {
        if (key == kPhotonCount) mPhotonCount = value;
        else if (key == kMaxBounces) mMaxBounces = value;
        else if (key == kGlobalRejectionProb) {
            mGlobalRejectionProb = value;
            FALCOR_ASSERT_GE(mGlobalRejectionProb, 0.0f);
            FALCOR_ASSERT_LT(mGlobalRejectionProb, 1.0f);
        }
        else if (key == kRussianRouletteWeight) {
            mRussianRouletteWeight = value;
            FALCOR_ASSERT_GT(mRussianRouletteWeight, 0.0f);
        }
        else if (key == kUseRussianRoulette) mUseRussianRoulette = value;
        else if (key == kUseWaveIntrinsics) mUseWaveIntrinsics = value;
        else logWarning("{} - Unknown property '{}'", getClassName(), key);
    }
}

Properties TracePhotons::getProperties() const
{
    Properties props;
    props[kPhotonCount] = mPhotonCount;
    props[kMaxBounces] = mMaxBounces;
    props[kGlobalRejectionProb] = mGlobalRejectionProb;
    props[kRussianRouletteWeight] = mRussianRouletteWeight;
    props[kUseRussianRoulette] = mUseRussianRoulette;
    props[kUseWaveIntrinsics] = mUseWaveIntrinsics;
    return props;
}

void TracePhotons::renderUI(Gui::Widgets& widget) {
    if (widget.var("Photon Count", mPhotonCount, 1u, 1u << 24u))
    {
        requestRecompile(); // Update buffer sizes
    }
    widget.var("Max Bounces", mMaxBounces, 1u, 20u);
    widget.var("Global Rejection Probability", mGlobalRejectionProb, 0.0f, 0.9f);
    widget.var("Russian Roulette Weight", mRussianRouletteWeight, 0.5f, 10.0f);
    widget.checkbox("Use Russian Roulette", mUseRussianRoulette);
    widget.checkbox("Use Wave Intrinsics", mUseWaveIntrinsics);
}

RenderPassReflection TracePhotons::reflect(const CompileData& compileData)
{
    // Define the required resources here
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

    FALCOR_ASSERT(mpPhotonSampler);

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    mpPhotonSampler->update(pRenderContext);

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.pProgram->addDefines(mpPhotonSampler->getDefines());
    mTracer.pProgram->addDefine("USE_RUSSIAN_ROULETTE", mUseRussianRoulette ? "1" : "0");
    mTracer.pProgram->addDefine("USE_WAVE_INTRINSICS", mUseWaveIntrinsics ? "1" : "0");

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
    var["CB"]["gGlobalSurvivalProb"] = (1.0f - mGlobalRejectionProb);
    var["CB"]["gInvSurvivalProb"] = 1.0f / (1.0f - mGlobalRejectionProb);
    var["CB"]["gPhotonHitCapacity"] = photonCapacity;
    var["CB"]["gRussianRouletteWeight"] = mRussianRouletteWeight;
    mpPhotonSampler->bindShaderData(var["gLights"]);

    const auto& pPhotonHits = renderData.getBuffer(kPhotonBuffer);
    FALCOR_ASSERT(pPhotonHits); // Do not need to clear because it will be overwritten
    var["gPhotonHits"] = pPhotonHits;

    const auto& pCounters = renderData.getBuffer(kCounterBuffer);
    FALCOR_ASSERT(pCounters);
    pCounters->setElement(0u, PhotonCounters{mPhotonCount, 0}); // Reset counters
    var["gCounters"] = pCounters;
    
    // Trace photons
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(mPhotonCount, 1, 1));

#ifdef _DEBUG
    // pRenderContext->uavBarrier(pCounters.get());
    const auto counters = pCounters->getElement<PhotonCounters>(0u);
    logInfo("photons={} emitted={} stored={}", mPhotonCount, counters.PhotonsEmitted, counters.PhotonStores);
#endif

    mFrameCount++;
}

void TracePhotons::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram.reset();
    mTracer.pBindingTable.reset();
    mTracer.pVars.reset();
    mFrameCount = 0;
    mpPhotonSampler.reset();

    // Set new scene.
    mpScene = pScene;

    if (mpScene)
    {
        mpPhotonSampler = PhotonSampler::create(pRenderContext, mpScene);

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
        mTracer.pProgram->addDefines(mpPhotonSampler->getDefines());
    }
}

void TracePhotons::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->addDefines(mpPhotonSampler->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}
