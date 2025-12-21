#include "Falcor.h"
#include "Core/Testbed.h"
#include "Utils/Scripting/Scripting.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <args.hxx>

using namespace Falcor;
using nlohmann::json;

const std::filesystem::path resultsDir = getProjectDirectory() / "results/";

// Helper to obtain (and create) a results directory, optionally nested.
std::filesystem::path getResultsDir(const std::string& subdir = "")
{
    std::filesystem::path path = resultsDir;
    if (!subdir.empty()) path /= subdir;
    std::filesystem::create_directories(path);
    return path;
}

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
ref<RenderGraph> graphSPPM(const ref<Device>& pDevice, bool reverseSearch = false, float rejProb = 0.0f, bool rr = true, bool stoch = true, bool reduction = true) {
    auto g = RenderGraph::create(pDevice, fmt::format("SPPM ({}{}, rej={}, rr={})", stoch ? "Stoch" : "", reverseSearch ? "PhotonSearch" : "QuerySearch", rejProb, rr, stoch));

    g->createPass("VisualizePhotons", "VisualizePhotons", Properties());
    g->createPass("TracePhotons", "TracePhotons", Properties(json {{"photonCount", 1<<20}, {"maxBounces", 8}, {"globalRejectionProb", rejProb}, {"useRussianRoulette", rr}}));
    Properties props = {json {{"visualizeHeatmap", false}, {"globalRadius", 0.02f}, {"causticRadius", 0.004f}, {"stochEval", stoch}, {"reverseSearch", reverseSearch}}};
    if (!reduction) {
        props["gloablAlpha"] = 1.0f;
        props["causticAlpha"] = 1.0f;
    }
    g->createPass("AccumPh", "AccumulatePhotonsRTX", props);
    g->createPass("TraceQueries", "TraceQueries", Properties(json {{"resetStatisticsPerFrame", false}}));

    for (const auto& output : g->getAvailableOutputs())
    {
        logInfo("RenderGraph output: {}", output);
    }
    for (const auto& [name, value] : g->getPassesDictionary())
    {
        // Avoid printing Dictionary::Value directly due to ambiguous conversions
        logInfo("{}", name);
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

ref<RenderGraph> graphPhotonNRCSameR(const ref<Device>& pDevice, float rej = 0.0f, bool stoch = true, bool reverse = false, float r = 0.015) {
    auto g = RenderGraph::create(pDevice, fmt::format("PhotonNRC (rej={}, stoch={}, rev={}, r={})", rej, stoch, reverse, r));

    g->createPass("TracePhotons", "TracePhotons", Properties(json {{"photonCount", 1<<19}, {"maxBounces", 8}, {"globalRejectionProb", rej}})); // OG used 1<<17
    g->createPass("AccumPh", "AccumulatePhotonsRTX", Properties(json {{"visualizeHeatmap", false}, {"globalRadius", r}, {"causticRadius", r}, {"stochEval", stoch}, {"reverseSearch", reverse}}));
    g->createPass("TraceQueries", "TraceQueries", Properties(json {{"resetStatisticsPerFrame", true}}));
    g->createPass("qsamp", "QuerySubsampling", Properties(json {{"count", 1<<15}, {"replacementFactor", 1.0f}})); // OG used 1<<17
    g->createPass("nrc", "NRC", Properties(json {{"jitFusion", true}}));

    g->addEdge("TraceQueries.queries", "qsamp.queries");
    g->addEdge("TraceQueries.nrcInput", "qsamp.nrcInput");

    g->addEdge("TracePhotons.photons", "AccumPh.photons");
    g->addEdge("TracePhotons.counters", "AccumPh.photonCounters");
    g->addEdge("qsamp.sample", "AccumPh.queries");

    g->addEdge("qsamp.nrcOutput", "nrc.trainInput");
    g->addEdge("AccumPh.outputBuffer", "nrc.trainTarget");
    g->addEdge("TraceQueries.nrcInput", "nrc.inferenceInput");
    g->addEdge("TraceQueries.queries", "nrc.inferenceQueries");

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

void captureOutputs(const ref<Testbed>& app, const ref<RenderGraph>& graph, const std::string& name = "", const std::filesystem::path& baseDir = resultsDir) {
    std::string outputName = name;
    if (outputName.empty()) {
        outputName = graph->getName();
    }
    std::filesystem::create_directories(baseDir);
    for (uint32_t i = 0; i < graph->getOutputCount(); ++i) {
        const auto pTex = graph->getOutput(i)->asTexture();
        if (!pTex) {
            logWarning("Output {} is not a texture, skipping capture.", graph->getOutputName(i));
            continue;
        }
        const auto fmt = pTex->getFormat();
        const auto isHdr = getFormatType(fmt) == FormatType::Float;
        const auto fileformat = isHdr ? Bitmap::FileFormat::ExrFile : Bitmap::FileFormat::PngFile;
        const auto ext = isHdr ? "exr" : "png";
        const auto path = baseDir / fmt::format("{}.{}.{}", outputName, graph->getOutputName(i), ext);
        pTex->captureToFile(0, 0, path, fileformat, Bitmap::ExportFlags::None);
        //app->captureOutput(resultsDir / (outputName + "." + graph->getOutputName(i) + ".exr"), i);
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
}

// Render for a fixed wall-clock duration with warmup frames first.
void renderForSeconds(const ref<Testbed>& app, const ref<RenderGraph>& graph, double seconds, uint32_t warmupFrames = 10)
{
    app->setRenderGraph(graph);

    for (uint32_t i = 0; i < warmupFrames; ++i) app->frame();

    auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds)
    {
        app->frame();
    }
}

ref<Testbed> createApp(const std::string& scene, uint32_t res = 512, bool interactive = false) {
    Testbed::Options options {};
    options.windowDesc.width = res;
    options.windowDesc.height = res;
    options.createWindow = interactive;

    auto app = Testbed::create(options);
    app->loadScene(scene);
    return app;
}

std::string getSceneName(const ref<Testbed>& app) {
    const auto scene = app->getScene()->getPath().stem().string();
    const auto res = app->getFrameBufferSize().x;
    return fmt::format("{}.{}", scene, res);
}

std::filesystem::path ensureReference(const ref<Testbed>& app) {
    std::filesystem::path path = resultsDir / fmt::format("ref.{}.exr", getSceneName(app));
    if (!std::filesystem::exists(path)) {
        auto pt = graphPT(app->getDevice());
        app->setRenderGraph(pt);
        for (uint32_t i = 0; i < 1<<13; ++i)
            app->frame();
        logInfo("Output format: {}", to_string(pt->getOutput(0)->asTexture()->getFormat()));
        pt->getOutput(0)->asTexture()->captureToFile(0, 0, path, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Uncompressed); // NOTE: Compression messes up loading later
    }
    return path;
}

ref<RenderGraph> graphFLIP(const ref<Device>& pDevice, const std::string& ref) {
    auto g = RenderGraph::create(pDevice, "FLIP");

    g->createPass("ref", "ImageLoader", Properties(json {{"filename", ref}, {"outputFormat", "RGBA32Float"}}));
    g->createPass("flip", "FLIPPass", Properties(json {{"isHDR", true}, {"computePooledFLIPValues", true}}));
    g->createPass("error", "ErrorMeasurePass", Properties(json {{"SelectedOutputId", "Difference"}}));
    g->createPass("tonemap", "ToneMapper", Properties());

    g->addEdge("ref.dst", "flip.referenceImage");
    g->addEdge("ref.dst", "error.Reference");
    g->markOutput("tonemap.dst");
    g->markOutput("flip.errorMapDisplay");
    g->markOutput("error.Output");
    return g;
}

namespace Falcor {
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Profiler::Stats, min, max, mean, stdDev)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Profiler::Capture::Lane, name, stats, records)
}

json getProperties(const ref<RenderGraph>& graph) {
    json j;
    for (const auto& pass : graph->getAllPasses()) {
        j[pass->getName()] = pass->getProperties().toJson();
    }
    return j;
}

void writeJson(const json& j, const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    file << std::setw(4) << j << std::endl;
}

void benchmarkQuality(const ref<Testbed>& app, const ref<RenderGraph>& graph, double seconds, const std::filesystem::path& dir) {
    auto ref = ensureReference(app);
    renderForSeconds(app, graph, seconds, 10);

    auto output = graph->getOutput(0);
    auto gFLIP = graphFLIP(app->getDevice(), ref);
    gFLIP->setInput("flip.testImage", output);
    gFLIP->setInput("error.Source", output);
    gFLIP->setInput("tonemap.src", output);
    app->setRenderGraph(gFLIP);
    app->frame();
    captureOutputs(app, gFLIP, fmt::format("{}.{}", graph->getName(), getSceneName(app)), dir);

    writeJson(getProperties(gFLIP), dir / fmt::format("{}.{}.json", graph->getName(), getSceneName(app)));
}

json benchmarkPerformance(const ref<Testbed>& app, const std::vector<ref<RenderGraph>>& graphs, uint32_t frames, uint32_t warmupFrames = 10) {
    json results;
    for (const auto& g : graphs) {
        app->setRenderGraph(g);

        for (uint32_t i = 0; i < warmupFrames; ++i) app->frame();

        app->getDevice()->getProfiler()->startCapture(frames);
        for (uint32_t i = 0; i < frames; ++i) app->frame();
        const auto capture = app->getDevice()->getProfiler()->endCapture();
        json info;
        for (const auto& lane : capture->getLanes()) {
            info[lane.name] = lane.stats;
        }
        info["name"] = g->getName();
        info["props"] = getProperties(g);

        results.push_back(info);
    }
    return results;
}

std::vector<uint2> getLinearResolutionLevels(uint max, uint n) {
    std::vector<uint2> levels;
    uint step = (max << 1);
    for (uint i = 1; i < n + 1; ++i) {
        uint pixels = step * i / n;
        uint res = pixels >> 1;
        levels.emplace_back(res, res);
    }
    return levels;
}

int runMain(int argc, char** argv)
{
    args::ArgumentParser parser("PhotonNRC Testbed");
    args::Flag sppmVsRes(parser, "sppm-res", "Run SPPM performance benchmark", {'s', "sppm-res"});
    args::Flag nrcVsRes(parser, "nrc-res", "Run PhotonNRC performance benchmark", {'n', "nrc-res"});
    args::Flag phnrcVsR(parser, "phnrc-r", "Run PhotonNRC vs radius performance benchmark", {'r', "phnrc-r"});
    args::Flag nrcVariants(parser, "nrc-variants", "Run NRC variants performance benchmark", {'v', "nrc-variants"});
    args::Flag quality(parser, "quality", "Run quality benchmark", {'q', "quality"});
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

    args::CompletionFlag completionFlag(parser, {"complete"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (const args::Completion& e)
    {
        std::cout << e.what();
        return 0;
    }
    catch (const args::Help&)
    {
        std::cout << parser;
        return 0;
    }
    catch (const args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (const args::RequiredError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    // Start Python interpreter
    Logger::setOutputs(Logger::OutputFlags::File | Logger::OutputFlags::Console);
    Scripting::start();
    // Register/load Falcor plugins so importers (e.g. .pyscene) are available.
    PluginManager::instance().loadAllPlugins();
    AssetResolver::getDefaultResolver().addSearchPath(getProjectDirectory() / "scenes", SearchPathPriority::First, AssetCategory::Scene);

    std::filesystem::create_directories(resultsDir);

    //auto app = createApp("cornell_box_caustic.pyscene", 512);
    // app->setRenderGraph(graphFLIP(app->getDevice(), ensureReference(app)));
    // app->run();
    //benchmarkQuality(app, graphSPPM(app->getDevice()), 512);

    if (args::get(sppmVsRes)) {
        json results = {};
        auto app = createApp("cornell_box_caustic.pyscene", 512);
        for (uint2 res : getLinearResolutionLevels(1080, 8)) {
            app->resizeFrameBuffer(res.x, res.y);
            results.push_back({
                {"resolution", {res.x, res.y}},
                {"MP", res.x * res.y / 1e6},
                {"runs", benchmarkPerformance(app, {
                    graphSPPM(app->getDevice(), false, 0.0f, true, false, false), // QuerySearch
                    graphSPPM(app->getDevice(), true, 0.0f, true, false, false),  // PhotonSearch
                    graphSPPM(app->getDevice(), false, 0.0f, true, true, false), // StochQuerySearch
                    graphSPPM(app->getDevice(), true, 0.0f, true, true, false),  // StochPhotonSearch
                    graphSPPM(app->getDevice(), false, 0.7f, true, false, false), // QuerySearch
                    graphSPPM(app->getDevice(), true, 0.7f, true, false, false),  // PhotonSearch
                    graphSPPM(app->getDevice(), false, 0.7f, true, true, false), // StochQuerySearch
                    graphSPPM(app->getDevice(), true, 0.7f, true, true, false),  // StochPhotonSearch
                }, 32)},
            });
        }
        writeJson(results, resultsDir / "sppm_vs_res.json");
    }

    if (args::get(nrcVsRes)) {
        json results = {};
        auto app = createApp("cornell_box_caustic.pyscene", 512);
        for (uint2 res : getLinearResolutionLevels(1920, 8)) {
            app->resizeFrameBuffer(res.x, res.y);
            results.push_back({
                {"resolution", {res.x, res.y}},
                {"MP", res.x * res.y / 1e6},
                {"runs", benchmarkPerformance(app, {
                    graphPT(app->getDevice()), // Path Tracing
                    graphPhotonNRC(app->getDevice(), 0.0f, false), // PhotonNRC
                    graphSPPM(app->getDevice(), false, 0.0f, true, true, false), // StochQuerySearch
                }, 32)},
            });
        }
        writeJson(results, resultsDir / "nrc_vs_res.json");
    }

    if (args::get(phnrcVsR)) {
        json results = {};
        auto app = createApp("cornell_box.pyscene", 512);
        for (float r : {0.001f, 0.005f, 0.01f, 0.015f, 0.02f, 0.03f}) {
            logInfo("Running PhotonNRC vs radius benchmark for r={}", r);
            results.push_back({
                {"r", r},
                {"runs", benchmarkPerformance(app, {
                    graphPhotonNRCSameR(app->getDevice(), 0.0f, true, false, r),
                    graphPhotonNRCSameR(app->getDevice(), 0.0f, true, true, r),
                    graphPhotonNRCSameR(app->getDevice(), 0.7f, true, false, r),
                    graphPhotonNRCSameR(app->getDevice(), 0.7f, true, true, r),
                }, 32)},
            });
        }
        writeJson(results, resultsDir / "phnrc_vs_r.json");
    }

    if (args::get(nrcVariants)) {
        auto app = createApp("cornell_box_caustic.pyscene", 512);
        auto json = benchmarkPerformance(app, {
            graphNRC(app->getDevice(), 1), // NRC
            graphNRC(app->getDevice(), 32), // MultiNRC
            graphBiNRC(app->getDevice()), // BiNRC
            graphPhotonNRC(app->getDevice()), // PhotonNRC
        }, 128);
        writeJson(json, resultsDir / "nrc_variants.json");
    }

    if (args::get(quality)) {
        for (const auto& scene : {
            "cornell_box_caustic.pyscene",
            "cornell_box.pyscene",
        }) {
            auto app = createApp(scene, 512);
            std::vector graphs = {
                graphNRC(app->getDevice(), 1), // NRC
                graphNRC(app->getDevice(), 32), // MultiNRC
                graphBiNRC(app->getDevice()), // BiNRC
                graphPhotonNRC(app->getDevice()), // PhotonNRC
                graphSPPM(app->getDevice()), // SPPM
            };
            for (const auto& g : graphs) {
                benchmarkQuality(app, g, 10.0, getResultsDir("quality"));
            }
        }
    }

    Scripting::shutdown();
    logInfo("Log file: {}", Logger::getLogFilePath());
    std::cout << "Log file: " << Logger::getLogFilePath() << std::endl;
    return 0;
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&] { return runMain(argc, argv); });
}