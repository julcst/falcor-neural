#include "Falcor.h"
#include "Core/Testbed.h"
#include "Utils/Scripting/Scripting.h"
#include <filesystem>

using namespace Falcor;
using nlohmann::json;

ref<RenderGraph> graphPT(const ref<Device>& pDevice) {
    auto g = RenderGraph::create(pDevice, "PT");

    g->createPass("accum", "AccumulatePass", Properties());
    g->createPass("pt", "PathTracer", Properties(json {{"maxDiffuseBounces", 8}, {"maxSpecularBounces", 8}}));
    g->createPass("vbuff", "VBufferRT", Properties());

    g->addEdge("vbuff.vbuffer", "pt.vbuffer");
    g->addEdge("vbuff.viewW", "pt.viewW");
    g->addEdge("vbuff.mvec", "pt.mvec");
    g->addEdge("pt.color", "accum.input");
    g->markOutput("accum.output");
    return g;
}

// NOTE: QuerySearch is faster, RR improves performance with minimal quality loss, rejProb speeds up even more with some quality loss
ref<RenderGraph> graphSPPM(const ref<Device>& pDevice, bool reverseSearch = false, float rejProb = 0.0f, bool rr = true, bool stoch = true) {
    auto g = RenderGraph::create(pDevice, fmt::format("SPPM ({}, rej={}, rr={}, stoch={})", reverseSearch ? "PhotonSearch" : "QuerySearch", rejProb, rr, stoch));

    g->createPass("VisualizePhotons", "VisualizePhotons", Properties());
    g->createPass("TracePhotons", "TracePhotons", Properties(json {{"photonCount", 1<<20}, {"maxBounces", 8}, {"globalRejectionProb", rejProb}, {"useRussianRoulette", rr}}));
    g->createPass("AccumPh", "AccumulatePhotonsRTX", Properties(json {{"visualizeHeatmap", false}, {"globalRadius", 0.01f}, {"causticRadius", 0.002f}, {"reverseSearch", reverseSearch}, {"stochEval", stoch}}));
    g->createPass("TraceQueries", "TraceQueries", Properties(json {{"resetStatisticsPerFrame", false}}));

    for (const auto& output : g->getAvailableOutputs())
    {
        logInfo("RenderGraph output: {}", output);
    }
    for (const auto& [name, value] : g->getPassesDictionary())
    {
        logInfo("{}:{}", name, value);
    }

    // VisualizePhotons
    g->addEdge("TracePhotons.photons", "VisualizePhotons.photons");
    g->addEdge("TracePhotons.counters", "VisualizePhotons.counters");

    // SPPM
    g->addEdge("TracePhotons.photons", "AccumPh.photons");
    g->addEdge("TracePhotons.counters", "AccumPh.photonCounters");
    g->addEdge("TraceQueries.queries", "AccumPh.queries");

    g->markOutput("AccumPh.outputTexture");
    //g->markOutput("VisualizePhotons");
    return g;
}

ref<RenderGraph> graphPhotonNRC(const ref<Device>& pDevice, float rej = 0.0f, bool stoch = true) {
    auto g = RenderGraph::create(pDevice, fmt::format("PhotonNRC (rej={}, stoch={})", rej, stoch));

    g->createPass("TracePhotons", "TracePhotons", Properties(json {{"photonCount", 1<<19}, {"maxBounces", 8}, {"globalRejectionProb", rej}})); // OG used 1<<17
    g->createPass("AccumPh", "AccumulatePhotonsRTX", Properties(json {{"visualizeHeatmap", false}, {"globalRadius", 0.015f}, {"causticRadius", 0.003f}, {"stochEval", stoch}}));
    g->createPass("Accum", "AccumulatePass", Properties());
    g->createPass("TraceQueries", "TraceQueries", Properties(json {{"resetStatisticsPerFrame", true}}));
    g->createPass("qsamp", "QuerySubsampling", Properties(json {{"count", 1<<15}, {"replacementFactor", 0.02f}})); // OG used 1<<17
    g->createPass("nrc", "NRC", Properties(json {{"jitFusion", true}}));
    g->createPass("visPh", "VisualizePhotons", Properties());
    g->createPass("debug", "DebugQueryBuffer", Properties());
    g->createPass("visQueries", "VisualizeQueries", Properties());

    g->addEdge("TracePhotons.photons", "visPh.photons");
    g->addEdge("TracePhotons.counters", "visPh.counters");
    // g->markOutput("visPh.dst");

    g->addEdge("TraceQueries.queries", "qsamp.queries");
    g->addEdge("TraceQueries.nrcInput", "qsamp.nrcInput");

    g->addEdge("TracePhotons.photons", "AccumPh.photons");
    g->addEdge("TracePhotons.counters", "AccumPh.photonCounters");
    g->addEdge("qsamp.sample", "AccumPh.queries");

    g->addEdge("qsamp.nrcOutput", "nrc.trainInput");
    g->addEdge("AccumPh.outputBuffer", "nrc.trainTarget");
    g->addEdge("TraceQueries.nrcInput", "nrc.inferenceInput");
    g->addEdge("TraceQueries.queries", "nrc.inferenceQueries");

    g->addEdge("TraceQueries.queries", "debug.queries");
    g->addEdge("TraceQueries.nrcInput", "debug.nrcInput");
    // g->markOutput("debug");

    g->addEdge("qsamp.sample", "visQueries.queries");
    g->addEdge("AccumPh.queryStates", "visQueries.queryStates");
    // g->markOutput("visQueries");

    g->markOutput("nrc.output");
    return g;
}

ref<RenderGraph> graphNRC(const ref<Device>& pDevice, uint32_t spp = 1) {
    auto g = RenderGraph::create(pDevice, spp == 1 ? "NRC" : fmt::format("MultisampleNRC (spp={})", spp));

    g->createPass("TraceQueries", "TraceQueries", Properties());
    g->createPass("qsamp", "QuerySubsampling", Properties(json {{"count", 1<<16}}));
    g->createPass("nrc", "NRC", Properties(json {{"jitFusion", true}}));
    g->createPass("PTQuery", "PathTracerQuery", Properties(json {{"maxDiffuseBounces", 8}, {"maxSpecularBounces", 8}, {"samplesPerPixel", spp}, {"parallelMultiSampling", true}}));

    g->addEdge("TraceQueries.queries", "qsamp.queries");
    g->addEdge("TraceQueries.nrcInput", "qsamp.nrcInput");

    g->addEdge("qsamp.sample", "PTQuery.queries");

    g->addEdge("qsamp.nrcOutput", "nrc.trainInput");
    g->addEdge("PTQuery.radiance", "nrc.trainTarget");
    g->addEdge("TraceQueries.nrcInput", "nrc.inferenceInput");
    g->addEdge("TraceQueries.queries", "nrc.inferenceQueries");

    g->markOutput("nrc.output");
    return g;
}

ref<RenderGraph> graphBiNRC(const ref<Device>& pDevice) {
    auto g = RenderGraph::create(pDevice, "BiNRC");

    g->createPass("TraceQueries", "TraceQueries", Properties());
    g->createPass("qsamp", "QuerySubsampling", Properties(json {{"count", 1<<16}}));
    g->createPass("nrc", "NRC", Properties(json {{"jitFusion", true}, {"useFactorization", false}})); // Factorization does not work with BiNRC
    g->createPass("estim", "PhotonNEE", Properties());

    g->addEdge("TraceQueries.queries", "qsamp.queries");
    g->addEdge("TraceQueries.nrcInput", "qsamp.nrcInput");

    g->addEdge("qsamp.sample", "estim.queries");

    g->addEdge("qsamp.nrcOutput", "nrc.trainInput");
    g->addEdge("estim.output", "nrc.trainTarget");
    g->addEdge("TraceQueries.nrcInput", "nrc.inferenceInput");
    g->addEdge("TraceQueries.queries", "nrc.inferenceQueries");

    g->markOutput("nrc.output");
    return g;
}

ref<RenderGraph> graphPTQuery(const ref<Device>& pDevice, uint32_t spp = 1) {
    auto g = RenderGraph::create(pDevice, fmt::format("PTQuery (spp={})", spp));

    g->createPass("TraceQueries", "TraceQueries", Properties());
    g->createPass("PTQuery", "PathTracerQuery", Properties(json {{"maxDiffuseBounces", 8}, {"maxSpecularBounces", 8}, {"samplesPerPixel", spp}, {"parallelMultiSampling", true}}));
    g->createPass("B2T", "BufferToTexture", Properties());

    g->addEdge("TraceQueries.queries", "PTQuery.queries");
    g->addEdge("PTQuery.radiance", "B2T.input");

    g->markOutput("B2T.output");
    return g;
}

std::vector<std::string> getMarkedOutputs(const ref<RenderGraph>& g) {
    std::vector<std::string> outputs;
    for (auto i = 0; i < g->getOutputCount(); ++i) {
        outputs.push_back(g->getOutputName(i));
    }
    return outputs;
}

std::string getMarkedOutput(const ref<RenderGraph>& g) {
    return g->getOutputName(0);
}

void addTonemapping(ref<RenderGraph>& g, const std::string& out) {
    g->createPass("tonemap", "ToneMapper", Properties());

    g->addEdge(out, "tonemap.src");
    g->markOutput("tonemap.dst");
}

void captureOutputs(const ref<Testbed>& app, const ref<RenderGraph>& graph, const std::string& prefix = "out_") {
    for (uint32_t i = 0; i < graph->getOutputCount(); ++i) {
        app->captureOutput(prefix + graph->getName() + "." + graph->getOutputName(i) + ".exr", i);
    }
}

void render(const ref<Testbed>& app, const ref<RenderGraph>& graph, uint32_t frameCount = 1, uint32_t warmupFrames = 10) {
    app->setRenderGraph(graph);

    logInfo("\nRender {} for {} frames\n===================================", graph->getName(), frameCount);
    for (uint32_t i = 0; i < warmupFrames && i < frameCount; ++i)
        app->frame();

    uint32_t profiledFrames = frameCount - warmupFrames;

    if (profiledFrames > 0) {
        app->getDevice()->getProfiler()->startCapture(profiledFrames);
        for (uint32_t i = 0; i < profiledFrames; ++i)
            app->frame();
        const auto capture = app->getDevice()->getProfiler()->endCapture();
        
        logInfo("\nStats for {} over {} frames\n===================================", graph->getName(), profiledFrames);
        for (const auto& lane : capture->getLanes()) {
            logInfo("{}: mean={} min={} max={} stdDev={}", lane.name, lane.stats.mean, lane.stats.min, lane.stats.max, lane.stats.stdDev);
        }
    }

    captureOutputs(app, graph);
}

ref<Testbed> createApp(const std::string& scene, uint32_t res = 512, bool interactive = false) {
    Testbed::Options options {};
    options.windowDesc.width = res;
    options.windowDesc.height = res;
    options.createWindow = interactive;
    options.colorFormat = ResourceFormat::RGBA32Float;

    auto app = Testbed::create(options);
    app->loadScene(scene);
    return app;
}

std::filesystem::path ensureReference(const ref<Testbed>& app) {
    const auto scene = app->getScene()->getPath().filename().string();
    const auto res = app->getFrameBufferSize().x;
    std::filesystem::path path = fmt::format("ref.{}.{}.exr", scene, res);
    if (!std::filesystem::exists(path)) {
        auto pt = graphPT(app->getDevice());
        app->setRenderGraph(pt);
        for (uint32_t i = 0; i < 1<<13; ++i)
            app->frame();
        logInfo("Output format: {}", to_string(pt->getOutput(0)->asTexture()->getFormat()));
        pt->getOutput(0)->asTexture()->captureToFile(0, 0, path, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Uncompressed); // NOTE: Compression messes up loading later
        //app->captureOutput(path);
    }
    return path;
}

ref<RenderGraph> graphFLIP(const ref<Device>& pDevice, const std::string& ref) {
    auto g = RenderGraph::create(pDevice, "FLIP");

    g->createPass("ref", "ImageLoader", Properties(json {{"filename", ref}}));
    g->createPass("flip", "FLIPPass", Properties(json {{"isHDR", true}}));
    g->createPass("error", "ErrorMeasurePass", Properties(json {{"SelectedOutputId", "Difference"}}));

    g->addEdge("ref.dst", "flip.referenceImage");
    g->addEdge("ref.dst", "error.Reference");
    g->markOutput("flip.errorMapDisplay");
    g->markOutput("error.Output");
    return g;
}

void benchmarkQuality(const ref<Testbed>& app, const ref<RenderGraph>& graph, uint32_t spp) {
    auto ref = ensureReference(app);
    app->setRenderGraph(graph);
    render(app, graph, spp);

    auto output = graph->getOutput(0);
    auto gFLIP = graphFLIP(app->getDevice(), ref);
    gFLIP->setInput("flip.testImage", output);
    gFLIP->setInput("error.Source", output);
    app->setRenderGraph(gFLIP);
    app->frame();
    captureOutputs(app, gFLIP);
}

int runMain(int argc, char** argv)
{
    // Start Python interpreter
    Logger::setOutputs(Logger::OutputFlags::File | Logger::OutputFlags::Console);
    Scripting::start();
    // Register/load Falcor plugins so importers (e.g. .pyscene) are available.
    PluginManager::instance().loadAllPlugins();
    AssetResolver::getDefaultResolver().addSearchPath(getProjectDirectory() / "scenes", SearchPathPriority::First, AssetCategory::Scene);

    auto app = createApp("cornell_box_caustic.pyscene", 512);
    // app->setRenderGraph(graphFLIP(app->getDevice(), ensureReference(app)));
    // app->run();
    benchmarkQuality(app, graphSPPM(app->getDevice()), 512);

    Scripting::shutdown();
    logInfo("Log file: {}", Logger::getLogFilePath());
    std::cout << "Log file: " << Logger::getLogFilePath() << std::endl;
    return 0;
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&] { return runMain(argc, argv); });
}