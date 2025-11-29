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

ref<RenderGraph> graphNRC(ref<Device> pDevice) {
    auto g = RenderGraph::create(pDevice, "NRC");

    g->createPass("TracePhotons", "TracePhotons", Properties(json {{"photonCount", 1<<23}}));
    g->createPass("AccumPh", "AccumulatePhotonsRTX", Properties(json {{"visualizeHeatmap", false}, {"radius", 0.005f}}));
    g->createPass("Accum", "AccumulatePass", Properties());
    g->createPass("TraceQueries", "TraceQueries", Properties());
    g->createPass("qsamp", "QuerySubsampling", Properties(json {{"count", 1<<14}}));
    g->createPass("nrc", "NRC", Properties());

    g->addEdge("TraceQueries.queries", "qsamp.queries");

    g->addEdge("TracePhotons.photons", "AccumPh.photons");
    g->addEdge("TracePhotons.counters", "AccumPh.photonCounters");
    g->addEdge("qsamp.sample", "AccumPh.queries");

    g->addEdge("qsamp.sample", "nrc.trainInput");
    g->addEdge("AccumPh.outputBuffer", "nrc.trainTarget");
    g->addEdge("TraceQueries.queries", "nrc.inferenceInput");

    g->markOutput("nrc.output");
    return g;
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
    // auto sppm = graphSPPM(app.getDevice());
    // app.setRenderGraph(sppm);
    // //app.getDevice()->getProfiler()->startCapture();
    // for (uint32_t i = 0; i < 256; ++i)
    //     app.frame();
    // //app.getDevice()->getProfiler()->endCapture()->writeToFile();
    // for (uint32_t i = 0; i < sppm->getOutputCount(); ++i)
    //     app.captureOutput("out_" + sppm->getOutputName(i) + ".exr", i);

    // NRC
    auto nrc = graphNRC(app.getDevice());
    app.setRenderGraph(nrc);
    //app.getDevice()->getProfiler()->startCapture();
    for (uint32_t i = 0; i < 256; ++i)
        app.frame();
    //app.getDevice()->getProfiler()->endCapture()->writeToFile();
    for (uint32_t i = 0; i < nrc->getOutputCount(); ++i)
        app.captureOutput("out_" + nrc->getOutputName(i) + ".exr", i);

    Scripting::shutdown();
    return 0;
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&] { return runMain(argc, argv); });
    logInfo("Log file: {}", Logger::getLogFilePath());
}