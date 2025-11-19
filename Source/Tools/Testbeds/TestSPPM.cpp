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
    g->markOutput("tonemap.dst");

    g->createPass("accum", "AccumulatePass", Properties());
    //g->addEdge("accum.output", "tonemap.src");

    // Reference PathTracer
    g->createPass("pt", "PathTracer", Properties());
    g->createPass("vbuff", "VBufferRT", Properties());
    g->addEdge("vbuff.vbuffer", "pt.vbuffer");
    g->addEdge("vbuff.viewW", "pt.viewW");
    g->addEdge("vbuff.mvec", "pt.mvec");
    g->addEdge("pt.color", "accum.input");

    // VisualizePhotons
    g->createPass("VisualizePhotons", "VisualizePhotons", Properties());
    g->addEdge("VisualizePhotons.dst", "tonemap.src");

    // TracePhotons
    g->createPass("TracePhotons", "TracePhotons", Properties());
    g->addEdge("TracePhotons.photons", "VisualizePhotons.photons");
    g->addEdge("TracePhotons.counters", "VisualizePhotons.counters");

    app.setRenderGraph(g);
    for (uint32_t i = 0; i < 2; ++i)
        app.frame();
    app.captureOutput("out.png");

    Scripting::shutdown();
    return 0;
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&] { return runMain(argc, argv); });
}