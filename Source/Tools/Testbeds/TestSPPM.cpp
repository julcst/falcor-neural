#include "Falcor.h"
#include "Core/Testbed.h"
#include "Utils/Scripting/Scripting.h"
#include <filesystem>

using namespace Falcor;

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

    if (!std::filesystem::exists("out_ref.exr")) {
        auto pt = graphPT(app.getDevice());
        app.setRenderGraph(pt);
        for (uint32_t i = 0; i < 1024; ++i)
            app.frame();
        app.captureOutput("out_ref.exr");
    }

    auto g = app.createRenderGraph("SPPM");

    g->createPass("Ref", "ImageLoader", Properties(nlohmann::json {{"filename", "out_ref.exr"}}));
    g->createPass("VisualizePhotons", "VisualizePhotons", Properties());
    g->createPass("TracePhotons", "TracePhotons", Properties());
    g->createPass("AccumPh", "AccumulatePhotonsRTX", Properties(nlohmann::json {{"visualizeHeatmap", false}}));
    g->createPass("Accum", "AccumulatePass", Properties());
    g->createPass("TraceQueries", "TraceQueries", Properties());
    g->createPass("Error", "ErrorMeasurePass", Properties(nlohmann::json  {{"SelectedOutputId", "Difference"}}));

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
    g->addEdge("AccumPh.output", "Accum.input");

    g->addEdge("Accum.output", "Error.Source");
    g->addEdge("Ref.dst", "Error.Reference");

    g->markOutput("AccumPh.output");
    g->markOutput("Accum.output");
    g->markOutput("VisualizePhotons.dst");
    g->markOutput("Error.Output");

    app.setRenderGraph(g);
    for (uint32_t i = 0; i < 32; ++i)
        app.frame();
    for (uint32_t i = 0; i < g->getOutputCount(); ++i)
        app.captureOutput("out_" + g->getOutputName(i) + ".exr", i);

    Scripting::shutdown();
    return 0;
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&] { return runMain(argc, argv); });
    logInfo("Log file: {}", Logger::getLogFilePath());
}