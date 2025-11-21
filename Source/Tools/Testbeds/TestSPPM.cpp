#include "Falcor.h"
#include "Core/Testbed.h"
#include "Utils/Scripting/Scripting.h"

using namespace Falcor;

int runMain(int argc, char** argv)
{
    // Start Python interprete
    Scripting::start();
    // Register/load Falcor plugins so importers (e.g. .pyscene) are available.
    PluginManager::instance().loadAllPlugins();

    Testbed::Options options {};
    options.windowDesc.width = 1080;
    options.windowDesc.height = 1080;
    Testbed app { options };
    app.loadScene("test_scenes/cornell_box.pyscene");

    //auto g = app.loadRenderGraph("scripts/SPPM.py");
    auto g = app.createRenderGraph("SPPM");

    g->createPass("tonemap", "ToneMapper", Properties());
    g->createPass("accum", "AccumulatePass", Properties());
    g->createPass("pt", "PathTracer", Properties());
    g->createPass("vbuff", "VBufferRT", Properties());
    g->createPass("VisualizePhotons", "VisualizePhotons", Properties());
    g->createPass("TracePhotons", "TracePhotons", Properties());
    g->createPass("accumph", "AccumulatePhotonsRTX", Properties());
    g->createPass("TraceQueries", "TraceQueries", Properties());

    // PathTracer
    g->addEdge("vbuff.vbuffer", "pt.vbuffer");
    g->addEdge("vbuff.viewW", "pt.viewW");
    g->addEdge("vbuff.mvec", "pt.mvec");
    g->addEdge("pt.color", "accum.input");

    // VisualizePhotons
    //g->addEdge("VisualizePhotons.dst", "tonemap.src");

    // TracePhotons
    for (const auto& output : g->getAvailableOutputs())
    {
        logInfo("RenderGraph output: {}", output);
    }
    for (const auto& [name, value] : g->getPassesDictionary())
    {
        logInfo("{}:{}", name, value);
    }
    
    g->addEdge("TracePhotons.photons", "VisualizePhotons.photons");
    g->addEdge("TracePhotons.counters", "VisualizePhotons.counters");
    g->addEdge("TracePhotons.photons", "accumph.photons");
    g->addEdge("TracePhotons.counters", "accumph.photonCounters");
    g->addEdge("TraceQueries.queries", "accumph.queries");
    g->addEdge("TraceQueries.aabbs", "accumph.queryAABBs");
    g->addEdge("accumph.output", "tonemap.src");

    //g->addEdge("accum.output", "tonemap.src");

    g->markOutput("tonemap.dst");

    app.setRenderGraph(g);
    for (uint32_t i = 0; i < 1; ++i)
        app.frame();
    app.captureOutput("out.png");

    Scripting::shutdown();
    return 0;
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&] { return runMain(argc, argv); });
}