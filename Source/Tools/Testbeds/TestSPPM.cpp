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
ref<RenderGraph> graphSPPM(const ref<Device>& pDevice, bool reverseSearch = false, float rejProb = 0.0f, bool rr = true) {
    auto g = RenderGraph::create(pDevice, fmt::format("SPPM ({}, rej={}, rr={})", reverseSearch ? "PhotonSearch" : "QuerySearch", rejProb, rr));

    g->createPass("Ref", "ImageLoader", Properties(json {{"filename", "out_ref.exr"}}));
    g->createPass("VisualizePhotons", "VisualizePhotons", Properties());
    g->createPass("TracePhotons", "TracePhotons", Properties(json {{"photonCount", 1<<20}, {"maxBounces", 5}, {"globalRejectionProb", rejProb}, {"useRussianRoulette", rr}}));
    g->createPass("AccumPh", "AccumulatePhotonsRTX", Properties(json {{"visualizeHeatmap", false}, {"globalRadius", 0.01f}, {"causticRadius", 0.002f}, {"reverseSearch", reverseSearch}}));
    g->createPass("Accum", "AccumulatePass", Properties());
    g->createPass("TraceQueries", "TraceQueries", Properties(json {{"resetStatisticsPerFrame", false}}));
    g->createPass("Error", "ErrorMeasurePass", Properties(json  {{"SelectedOutputId", "Difference"}}));

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

    g->addEdge("AccumPh.outputTexture", "Error.Source");
    g->addEdge("Ref.dst", "Error.Reference");

    g->markOutput("AccumPh.outputTexture");
    g->markOutput("VisualizePhotons.dst");
    //g->markOutput("Error.Output");
    return g;
}

ref<RenderGraph> graphPhotonNRC(const ref<Device>& pDevice, float rej = 0.0f) {
    auto g = RenderGraph::create(pDevice, fmt::format("PhotonNRC (rej={})", rej));

    g->createPass("TracePhotons", "TracePhotons", Properties(json {{"photonCount", 1<<18}, {"maxBounces", 6}, {"globalRejectionProb", rej}})); // OG used 1<<17
    g->createPass("AccumPh", "AccumulatePhotonsRTX", Properties(json {{"visualizeHeatmap", false}, {"globalRadius", 0.015f}, {"causticRadius", 0.003f}}));
    g->createPass("Accum", "AccumulatePass", Properties());
    g->createPass("TraceQueries", "TraceQueries", Properties(json {{"resetStatisticsPerFrame", true}}));
    g->createPass("qsamp", "QuerySubsampling", Properties(json {{"count", 1<<14}, {"replacementFactor", 0.02f}})); // OG used 1<<17
    g->createPass("nrc", "NRC", Properties());
    g->createPass("visPh", "VisualizePhotons", Properties());
    g->createPass("debug", "DebugQueryBuffer", Properties());
    g->createPass("visQueries", "VisualizeQueries", Properties());

    g->addEdge("TracePhotons.photons", "visPh.photons");
    g->addEdge("TracePhotons.counters", "visPh.counters");
    g->markOutput("visPh.dst");

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
    // g->markOutput("debug.queryPosition");
    // g->markOutput("debug.queryThroughput");
    // g->markOutput("debug.queryEmission");
    // g->markOutput("debug.nrcDiffuse");
    // g->markOutput("debug.nrcWo");
    // g->markOutput("debug.queryNormal");
    // g->markOutput("debug.nrcRoughness");

    g->addEdge("qsamp.sample", "visQueries.queries");
    g->addEdge("AccumPh.queryStates", "visQueries.queryStates");
    g->markOutput("visQueries.output");

    g->markOutput("nrc.output");
    return g;
}

ref<RenderGraph> graphNRC(const ref<Device>& pDevice, uint32_t spp = 1) {
    auto g = RenderGraph::create(pDevice, spp == 1 ? "NRC" : fmt::format("MultisampleNRC (spp={})", spp));

    g->createPass("TraceQueries", "TraceQueries", Properties());
    g->createPass("qsamp", "QuerySubsampling", Properties(json {{"count", 1<<16}}));
    g->createPass("nrc", "NRC", Properties());
    g->createPass("PTQuery", "PathTracerQuery", Properties(json {{"maxDiffuseBounces", 8}, {"maxSpecularBounces", 8}, {"samplesPerPixel", spp}}));

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
    g->createPass("nrc", "NRC", Properties());
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
    g->createPass("PTQuery", "PathTracerQuery", Properties(json {{"maxDiffuseBounces", 8}, {"maxSpecularBounces", 8}, {"samplesPerPixel", spp}}));
    g->createPass("B2T", "BufferToTexture", Properties());

    g->addEdge("TraceQueries.queries", "PTQuery.queries");
    g->addEdge("PTQuery.radiance", "B2T.input");

    g->markOutput("B2T.output");
    return g;
}

void captureOutputs(Testbed& app, const ref<RenderGraph>& graph, const std::string& prefix = "out_") {
    for (uint32_t i = 0; i < graph->getOutputCount(); ++i) {
        app.captureOutput(prefix + graph->getName() + "." + graph->getOutputName(i) + ".exr", i);
    }
}

void render(Testbed& app, const ref<RenderGraph>& graph, uint32_t frameCount = 1, uint32_t warmupFrames = 10) {
    app.setRenderGraph(graph);

    logInfo("\nRender {} for {} frames\n===================================", graph->getName(), frameCount);
    for (uint32_t i = 0; i < warmupFrames; ++i)
        app.frame();

    uint32_t profiledFrames = frameCount - warmupFrames;
    app.getDevice()->getProfiler()->startCapture(profiledFrames);
    for (uint32_t i = 0; i < profiledFrames; ++i)
        app.frame();
    const auto capture = app.getDevice()->getProfiler()->endCapture();

    if (profiledFrames > 0) {
        logInfo("\nStats for {} over {} frames\n===================================", graph->getName(), profiledFrames);
        for (const auto& lane : capture->getLanes()) {
            logInfo("{}: mean={} min={} max={} stdDev={}", lane.name, lane.stats.mean, lane.stats.min, lane.stats.max, lane.stats.stdDev);
        }
    }

    captureOutputs(app, graph);
}

int runMain(int argc, char** argv)
{
    // Start Python interprete
    Scripting::start();
    // Register/load Falcor plugins so importers (e.g. .pyscene) are available.
    PluginManager::instance().loadAllPlugins();

    const uint32_t res = 512;
    Testbed::Options options {};
    options.windowDesc.width = res;
    options.windowDesc.height = res;
    // options.createWindow = true; // Toggle preview
    Testbed app { options };
    AssetResolver::getDefaultResolver().addSearchPath(getProjectDirectory() / "scenes", SearchPathPriority::First, AssetCategory::Scene);
    app.loadScene("cornell_box.pyscene");

    // Preview
    if (options.createWindow) {
        auto pt = graphPhotonNRC(app.getDevice());
        app.setRenderGraph(pt);
        app.frame();
        app.getDevice()->getProfiler()->startCapture();
        app.run();
        return 0;
    }

    // Reference PT
    if (!std::filesystem::exists("out_ref.exr")) {
        auto pt = graphPT(app.getDevice());
        app.setRenderGraph(pt);
        for (uint32_t i = 0; i < 1<<13; ++i)
            app.frame();
        app.captureOutput("out_ref.exr");
    }

    // SPPM
    // render(app, graphSPPM(app.getDevice(), false, 0.5f, false), 512);
    render(app, graphSPPM(app.getDevice(), false, 0.5f, true), 512);
    //render(app, graphSPPM(app.getDevice(), true), 32);

    // PhotonNRC
    render(app, graphPhotonNRC(app.getDevice()), 256);
    render(app, graphPhotonNRC(app.getDevice(), 0.7f), 256);

    // NRC
    render(app, graphNRC(app.getDevice()), 128);

    // PhotonNEE
    render(app, graphBiNRC(app.getDevice()), 128);

    // Multisample NRC
    // render(app, graphNRC(app.getDevice(), 32), 128);

    // PT Query
    // render(app, graphPTQuery(app.getDevice()));
    // render(app, graphPTQuery(app.getDevice(), 32));

    Scripting::shutdown();
    logInfo("Log file: {}", Logger::getLogFilePath());
    return 0;
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&] { return runMain(argc, argv); });
}