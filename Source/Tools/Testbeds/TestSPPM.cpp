#include "Falcor.h"
#include "Core/Testbed.h"
#include "Utils/Scripting/Scripting.h"
#include <filesystem>

using namespace Falcor;
using nlohmann::json;

ref<RenderGraph> graphPT(ref<Device> pDevice) {
    auto g = RenderGraph::create(pDevice, "PT");

    g->createPass("accum", "AccumulatePass", Properties());
    g->createPass("pt", "PathTracer", Properties());
    g->createPass("vbuff", "VBufferRT", Properties());

    g->addEdge("vbuff.vbuffer", "pt.vbuffer");
    g->addEdge("vbuff.viewW", "pt.viewW");
    g->addEdge("vbuff.mvec", "pt.mvec");
    g->addEdge("pt.color", "accum.input");
    g->markOutput("accum.output");
    return g;
}

ref<RenderGraph> graphSPPM(ref<Device> pDevice) {
    auto g = RenderGraph::create(pDevice, "SPPM");

    g->createPass("Ref", "ImageLoader", Properties(json {{"filename", "out_ref.exr"}}));
    g->createPass("VisualizePhotons", "VisualizePhotons", Properties());
    g->createPass("TracePhotons", "TracePhotons", Properties(json {{"photonCount", 2<<22}}));
    g->createPass("AccumPh", "AccumulatePhotonsRTX", Properties(json {{"visualizeHeatmap", false}, {"radius", 0.005f}}));
    g->createPass("Accum", "AccumulatePass", Properties());
    g->createPass("TraceQueries", "TraceQueries", Properties());
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
    g->markOutput("Error.Output");
    return g;
}

ref<RenderGraph> graphPhotonNRC(ref<Device> pDevice) {
    auto g = RenderGraph::create(pDevice, "PhtonNRC");

    g->createPass("TracePhotons", "TracePhotons", Properties(json {{"photonCount", 1<<23}}));
    g->createPass("AccumPh", "AccumulatePhotonsRTX", Properties(json {{"visualizeHeatmap", false}, {"radius", 0.005f}}));
    g->createPass("Accum", "AccumulatePass", Properties());
    g->createPass("TraceQueries", "TraceQueries", Properties());
    g->createPass("qsamp", "QuerySubsampling", Properties(json {{"count", 1<<14}}));
    g->createPass("nrc", "NRC", Properties());
    g->createPass("visPh", "VisualizePhotons", Properties());

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

    g->markOutput("nrc.output");
    return g;
}

ref<RenderGraph> graphNRC(ref<Device> pDevice) {
    auto g = RenderGraph::create(pDevice, "NRC");

    g->createPass("TraceQueries", "TraceQueries", Properties());
    g->createPass("qsamp", "QuerySubsampling", Properties(json {{"count", 1<<14}}));
    g->createPass("nrc", "NRC", Properties());
    g->createPass("PTQuery", "PathTracerQuery", Properties());

    g->addEdge("TraceQueries.queries", "qsamp.queries");
    g->addEdge("TraceQueries.nrcInput", "qsamp.nrcInput");

    g->addEdge("qsamp.sample", "PTQuery.queries");

    g->addEdge("qsamp.nrcOutput", "nrc.trainInput");
    g->addEdge("PTQuery.radiance", "nrc.trainTarget");
    g->addEdge("TraceQueries.nrcInput", "nrc.inferenceInput");

    g->markOutput("nrc.output");
    return g;
}

ref<RenderGraph> graphPTQuery(ref<Device> pDevice) {
    auto g = RenderGraph::create(pDevice, "PTQuery");

    g->createPass("TraceQueries", "TraceQueries", Properties());
    g->createPass("PTQuery", "PathTracerQuery", Properties());
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

void render(Testbed& app, const ref<RenderGraph>& graph, uint32_t frameCount = 1) {
    app.setRenderGraph(graph);
    //app.getDevice()->getProfiler()->startCapture();
    for (uint32_t i = 0; i < frameCount; ++i)
        app.frame();
    //app.getDevice()->getProfiler()->endCapture()->writeToFile();
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
    Testbed app { options };
    app.loadScene("test_scenes/cornell_box.pyscene");

    // Reference PT
    if (!std::filesystem::exists("out_ref.exr")) {
        auto pt = graphPT(app.getDevice());
        app.setRenderGraph(pt);
        for (uint32_t i = 0; i < 1024; ++i)
            app.frame();
        app.captureOutput("out_ref.exr");
    }

    // SPPM
    // render(app, graphSPPM(app.getDevice()), 128);

    // PhotonNRC
    // render(app, graphPhotonNRC(app.getDevice()), 128);

    // NRC
    render(app, graphNRC(app.getDevice()), 128);

    // // PT Query
    //render(app, graphPTQuery(app.getDevice()));

    Scripting::shutdown();
    logInfo("Log file: {}", Logger::getLogFilePath());
    return 0;
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&] { return runMain(argc, argv); });
}