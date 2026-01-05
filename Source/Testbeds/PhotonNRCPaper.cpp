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
using GraphConfigurator = std::function<ref<RenderGraph>(const ref<RenderGraph>&)>;

namespace Falcor {
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Profiler::Stats, min, max, mean, stdDev)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Profiler::Capture::Lane, name, stats, records)
}

const std::filesystem::path resultsDir = getProjectDirectory() / "results/";

// Helper to obtain (and create) a results directory, optionally nested.
std::filesystem::path getResultsDir(const std::string& subdir = "")
{
    std::filesystem::path path = resultsDir;
    if (!subdir.empty()) path /= subdir;
    std::filesystem::create_directories(path);
    return path;
}

GraphConfigurator graphPT() {
    return [](const ref<RenderGraph>& g) {
        g->setName("PT");
        g->createPass("accum", "AccumulatePass", Properties());
        g->createPass("pt", "PathTracer", Properties(json {{"maxDiffuseBounces", 8}, {"maxSpecularBounces", 8}}));
        g->createPass("vbuff", "VBufferRT", Properties());

        g->addEdge("vbuff.vbuffer", "pt.vbuffer");
        g->addEdge("vbuff.viewW", "pt.viewW");
        g->addEdge("vbuff.mvec", "pt.mvec");
        g->addEdge("pt.color", "accum.input");
        g->markOutput("accum.output");
        return g;
    };
}

// NOTE: QuerySearch is faster, RR improves performance with minimal quality loss, rejProb speeds up even more with some quality loss
GraphConfigurator graphSPPM(bool reverseSearch = false, float rejProb = 0.0f, bool rr = true, bool stoch = true, bool reduction = true) {
    return [=](const ref<RenderGraph>& g) {
        g->setName("SPPM");
        g->createPass("VisualizePhotons", "VisualizePhotons", Properties());
        // NOTE: photon count > 2e12 crashes PhotonSearch on Blackwell with Xid 109: CTX SWITCH TIMEOUT
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

        // g->createPass("debug", "DebugQueryBuffer", Properties());
        // g->addEdge("TraceQueries.queries", "debug.queries");
        // g->markOutput("debug.queryGeomID");

        g->markOutput("AccumPh.outputTexture");
        //g->markOutput("VisualizePhotons");
        return g;
    };
}

GraphConfigurator graphNRCSPPC(float rej = 0.0f, bool stoch = true) {
    return [=](const ref<RenderGraph>& g) {
        g->setName("NRC+SPPC");
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
    };
}

GraphConfigurator graphNRCSPPCSameR(float rej = 0.0f, bool stoch = true, bool reverse = false, float r = 0.015) {
    return [=](const ref<RenderGraph>& g) {
        g->setName("NRC+SPPC");
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
    };
}

GraphConfigurator graphNRCPT(uint32_t spp = 1) {
    return [=](const ref<RenderGraph>& g) {
        g->setName(spp == 1 ? "NRC+PT" : fmt::format("NRC+PT{}", spp));
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
    };
}

GraphConfigurator graphNRCLT(uint32_t maxBounces = 6, bool visualizeQueries = false, uint32_t mode = 0) {
    return [=](const ref<RenderGraph>& g) {
        g->setName(mode == 0 ? "NRC+LT" : (mode == 1 ? "NRC+WarpLT" : "NRC+ResLT"));
        g->createPass("TraceQueries", "TraceQueries", Properties(json {{"terminateDiffuse", false}, {"terminateBTH", true}}));
        g->createPass("qsamp", "QuerySubsampling", Properties(json {{"count", 1<<16}}));
        g->createPass("nrc", "NRC", Properties(json {{"jitFusion", true}, {"useFactorization", true}})); // Factorization does not work with BiNRC
        g->createPass("estim", "PhotonNEE", Properties(json {{"maxBounces", maxBounces}, {"mode", mode}}));

        g->addEdge("TraceQueries.queries", "qsamp.queries");
        g->addEdge("TraceQueries.nrcInput", "qsamp.nrcInput");

        g->addEdge("qsamp.sample", "estim.queries");

        g->addEdge("qsamp.nrcOutput", "nrc.trainInput");
        g->addEdge("estim.output", "nrc.trainTarget");
        g->addEdge("TraceQueries.nrcInput", "nrc.inferenceInput");
        g->addEdge("TraceQueries.queries", "nrc.inferenceQueries");

        if (visualizeQueries)
        {
            g->createPass("visQueries", "VisualizeQueries", Properties(json {{"constantRadius", 0.005f}, {"mode", 3}}));
            g->addEdge("qsamp.sample", "visQueries.queries");
            g->addEdge("estim.output", "visQueries.colors");
            g->markOutput("visQueries.output");
        }

        g->markOutput("nrc.output");
        return g;
    };
}

GraphConfigurator graphPTQuery(uint32_t spp = 1) {
    return [=](const ref<RenderGraph>& g) {
        g->setName("PT");
        g->createPass("TraceQueries", "TraceQueries", Properties());
        g->createPass("PTQuery", "PathTracerQuery", Properties(json {{"maxDiffuseBounces", 8}, {"maxSpecularBounces", 8}, {"samplesPerPixel", spp}, {"parallelMultiSampling", true}}));
        g->createPass("B2T", "BufferToTexture", Properties());

        g->addEdge("TraceQueries.queries", "PTQuery.queries");
        g->addEdge("PTQuery.radiance", "B2T.input");

        g->markOutput("B2T.output");
        return g;
    };
}

GraphConfigurator graphTonemap() {
    return [](const ref<RenderGraph>& g) {
        g->setName("Tonemap");
        g->createPass("tonemap", "ToneMapper", Properties());
        g->markOutput("tonemap.dst");
        return g;
    };
}

GraphConfigurator graphFLIP(const std::string& refPath) {
    return [=](const ref<RenderGraph>& g) {
        g->setName("FLIP");
        g->createPass("ref", "ImageLoader", Properties(json {{"filename", refPath}, {"outputFormat", "RGBA32Float"}}));
        g->createPass("flip", "FLIPPass", Properties(json {{"isHDR", true}, {"computePooledFLIPValues", true}}));
        g->createPass("error", "ErrorMeasurePass", Properties(json {{"SelectedOutputId", "Difference"}}));
        g->createPass("tonemap", "ToneMapper", Properties());

        g->addEdge("ref.dst", "flip.referenceImage");
        g->addEdge("ref.dst", "error.Reference");
        g->markOutput("tonemap.dst");
        g->markOutput("flip.errorMapDisplay");
        g->markOutput("error.Output");
        return g;
    };
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

// Compose a configurator with a name change
GraphConfigurator rename(GraphConfigurator config, const std::string& name) {
    return [=](const ref<RenderGraph>& g) {
        auto result = config(g);
        result->setName(name);
        return result;
    };
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
        logInfo("Captured output {} to {} ({})", graph->getOutputName(i), path.string(), to_string(fmt));
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
// Returns JSON with elapsed time, frame count, profiling data, and other metrics.
json renderForSeconds(const ref<Testbed>& app, const ref<RenderGraph>& graph, double seconds, uint32_t warmupFrames = 10)
{
    app->setRenderGraph(graph);
    
    for (uint32_t i = 0; i < warmupFrames; ++i) app->frame();

    app->getDevice()->getProfiler()->startCapture();
    
    auto start = std::chrono::steady_clock::now();
    uint32_t frameCount = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds)
    {
        app->frame();
        frameCount++;
    }
    
    double elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    
    auto capture = app->getDevice()->getProfiler()->endCapture();
    
    json result;
    result["elapsedSeconds"] = elapsedSeconds;
    result["frameCount"] = frameCount;
    result["warmupFrames"] = warmupFrames;
    
    // Collect profiling data if enabled
    json profileStats;
    for (const auto& lane : capture->getLanes()) {
        profileStats[lane.name] = lane.stats;
    }
    result["profile"] = profileStats;
    
    return result;
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

// Render PT until convergence by measuring frame-to-frame difference.
// Returns the render graph with the converged texture as output.
// errorThreshold: stop when frame difference is below this
// maxMinutes: maximum time in minutes to spend rendering
// batchSize: check convergence and log every batchSize samples
ref<RenderGraph> renderPTUntilConvergence(const ref<Testbed>& app, float errorThreshold = 0.01f, double maxMinutes = 5.0, uint32_t batchSize = 128) {
    auto pt = graphPT()(app->createRenderGraph());
    app->setRenderGraph(pt);
    
    // Create error measure graph to compare consecutive frames
    auto gError = RenderGraph::create(app->getDevice(), "PTConvergenceCheck");
    gError->createPass("error", "ErrorMeasurePass", Properties(json {{"SelectedOutputId", "Difference"}, {"ComputeSquaredDifference", true}, {"ComputeAverage", true}}));
    gError->markOutput("error.Output");
    
    uint32_t spp = 0;
    const uint32_t samplesPerFrame = 1;
    ref<Texture> prevFrame = nullptr;
    ref<Texture> currentFrame = nullptr;
    
    logInfo("Rendering PT until convergence (error threshold={}, max time={} minutes, batch size={})", errorThreshold, maxMinutes, batchSize);
    
    auto startTime = std::chrono::steady_clock::now();
    double maxSeconds = maxMinutes * 60.0;
    
    while (true) {
        // Check time limit
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
        if (elapsed > maxSeconds) {
            logInfo("Time limit reached ({} minutes)", maxMinutes);
            break;
        }
        
        // Render frames in a batch
        for (uint32_t i = 0; i < batchSize; ++i) {
            app->setRenderGraph(pt);
            app->frame();
            if (i == batchSize - 2) prevFrame = pt->getOutput(0)->asTexture();
            if (i == batchSize - 1) currentFrame = pt->getOutput(0)->asTexture();
            spp += samplesPerFrame;
        }
        
        // Check convergence at batch boundaries
        if (prevFrame) {
            gError->setInput("error.Reference", prevFrame);
            gError->setInput("error.Source", currentFrame);
            app->setRenderGraph(gError);
            app->frame();
            
            // Get MSE from error pass properties
            auto errorPass = gError->getPass("error");
            float avgError = errorPass->getProperties()["avgError"];
            
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
            logInfo("SPP: {}, Frame-to-Frame Error: {}, Elapsed: {:.1f}s", spp, avgError, elapsed);
            
            if (avgError < errorThreshold) {
                logInfo("Converged at {} spp with error {} after {:.1f}s", spp, avgError, elapsed);
                break;
            }
        }

        prevFrame = currentFrame;
    }
    
    logInfo("Rendering complete at {} spp", spp);
    return pt;
}

std::filesystem::path ensureReference(const ref<Testbed>& app) {
    std::filesystem::path refDir = resultsDir / "ref";
    std::filesystem::path path = refDir / fmt::format("{}.exr", getSceneName(app));
    if (!std::filesystem::exists(path)) {
        logInfo("Building reference image: {}", path);
        auto ptGraph = renderPTUntilConvergence(app, 1e-8f, 30.0); // 30 minutes, very low error threshold
        
        auto convergedTex = ptGraph->getOutput(0)->asTexture();
        logInfo("Output format: {}", to_string(convergedTex->getFormat()));
        convergedTex->captureToFile(0, 0, path, Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Uncompressed); // NOTE: Compression messes up loading later

        auto tm = graphTonemap()(app->createRenderGraph());
        tm->setInput("tonemap.src", convergedTex);
        app->setRenderGraph(tm);
        app->frame();
        captureOutputs(app, tm, getSceneName(app), refDir);
    }
    return path;
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

void benchmarkQuality(const ref<Testbed>& app, const ref<RenderGraph>& graph, double seconds, const std::filesystem::path& dir, uint32_t warmupFrames = 10) {
    auto ref = ensureReference(app);
    
    // Render and profile (includes warmup and profiling)
    auto renderStats = renderForSeconds(app, graph, seconds, warmupFrames);

    auto output = graph->getOutput(0);
    auto gFLIP = graphFLIP(ref)(app->createRenderGraph());
    gFLIP->setInput("flip.testImage", output);
    gFLIP->setInput("error.Source", output);
    gFLIP->setInput("tonemap.src", output);
    app->setRenderGraph(gFLIP);
    app->frame();
    
    // Capture outputs
    captureOutputs(app, gFLIP, fmt::format("{}.{}", graph->getName(), getSceneName(app)), dir);

    // Collect FLIP metrics and combine with profiling stats
    json outputJson = getProperties(graph);
    outputJson.merge_patch(getProperties(gFLIP));
    outputJson["renderStats"] = renderStats;
    
    writeJson(outputJson, dir / fmt::format("{}.{}.json", graph->getName(), getSceneName(app)));
}

json benchmarkPerformance(const ref<Testbed>& app, const std::vector<GraphConfigurator>& graphConfigs, uint32_t frames, uint32_t warmupFrames = 10) {
    json results;
    for (const auto& config : graphConfigs) {
        auto g = config(app->createRenderGraph());
        logInfo("Running performance test for: {}", g->getName());
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

json benchmarkConvergence(const ref<Testbed>& app, const std::vector<GraphConfigurator>& graphConfigs, const std::filesystem::path& referencePath, double timeMinutes, uint32_t warmup = 4) {
    json results;
    
    // Load reference once for all graphs
    auto refTexture = Texture::createFromFile(app->getDevice(), referencePath, false, false);
    
    for (const auto& config : graphConfigs) {
        auto g = config(app->createRenderGraph());
        logInfo("Running convergence test for: {}", g->getName());
        
        // Add FLIP and error passes to the graph
        g->createPass("refLoader", "ImageLoader", Properties(json {{"filename", referencePath.string()}, {"outputFormat", "RGBA32Float"}}));
        g->createPass("flip", "FLIPPass", Properties(json {{"isHDR", true}, {"computePooledFLIPValues", true}}));
        g->createPass("error", "ErrorMeasurePass", Properties(json {{"ComputeSquaredDifference", true}, {"ComputeAverage", true}}));
        
        // Get the main output
        std::string mainOutput = g->getOutputName(0);
        
        // Connect FLIP and error passes
        g->addEdge("refLoader.dst", "flip.referenceImage");
        g->addEdge(mainOutput, "flip.testImage");
        g->addEdge("refLoader.dst", "error.Reference");
        g->addEdge(mainOutput, "error.Source");
        
        g->markOutput("flip.errorMap");
        g->markOutput("error.Output");
        
        json graphData;
        graphData["name"] = g->getName();
        graphData["properties"] = getProperties(g);
        json measurements = json::array();
        
        app->setRenderGraph(g);
        
        auto startTime = std::chrono::steady_clock::now();
        double maxSeconds = timeMinutes * 60.0;
        uint32_t frameCount = 0;
        
        while (true) {
            auto elapsed = frameCount < warmup ? 0.0 : std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();

            if (elapsed > maxSeconds) {
                break;
            }
            
            // Render frame (includes FLIP and error computation)
            app->frame();
            frameCount++;
            
            // Get metrics from passes
            auto flipJson = g->getPass("flip")->getProperties().toJson();
            auto errorJson = g->getPass("error")->getProperties().toJson();

            double min = flipJson["minFLIP"];
            double max = flipJson["maxFLIP"];
            double avg = flipJson["averageFLIP"];
            double mse = errorJson["avgError"];

            logInfo("Frame {}: Time {:.1f}s, FLIP avg {}, min {}, max {}, MSE {}", 
                    frameCount, elapsed, avg, min, max, mse);

            measurements.push_back({
                {"frame", frameCount},
                {"time", elapsed},
                {"flip", avg},
                {"min", min},
                {"max", max},
                {"mse", mse}
            });
        }
        
        graphData["measurements"] = measurements;
        graphData["totalFrames"] = frameCount;
        graphData["totalTime"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();
        results.push_back(graphData);
        
        logInfo("Completed convergence test for {} ({} frames in {:.1f}s)", g->getName(), frameCount, (double)graphData["totalTime"]);
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
    args::Flag nrcVsRes(parser, "nrc-res", "Run NRC performance benchmark", {'n', "nrc-res"});
    args::Flag phnrcVsR(parser, "phnrc-r", "Run NRC+SPPC vs radius performance benchmark", {'r', "phnrc-r"});
    args::Flag nrcVariants(parser, "nrc-variants", "Run NRC variants performance benchmark", {'v', "nrc-variants"});
    args::Flag quality(parser, "quality", "Run quality benchmark", {'q', "quality"});
    args::Flag buildRef(parser, "build-ref", "Build all reference images", {'b', "build-ref"});
    args::Flag convergenceTest(parser, "convergence", "Run convergence test comparing NRC variants", {'c', "convergence"});
    args::Flag ltTest(parser, "lt-test", "Run NRC+LT test", {"lt", "lt-test"});
    args::Flag sppmTest(parser, "sppm-test", "Run SPPM test", {"sppm", "sppm-test"});
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

    if (args::get(buildRef)) {
        for (const auto& scene : {
            "cornell_box.pyscene",
            "cornell_box_caustic.pyscene",
            "cornell_box_bunny.pyscene",
        }) {
            auto app = createApp(scene, 512);
            logInfo("Building reference for scene: {}", scene);
            ensureReference(app);
        }
        logInfo("All reference images built successfully.");
        Scripting::shutdown();
        logInfo("Log file: {}", Logger::getLogFilePath());
        std::cout << "Log file: " << Logger::getLogFilePath() << std::endl;
        return 0;
    }

    if (args::get(sppmVsRes)) {
        json results = {};
        auto app = createApp("cornell_box_caustic.pyscene", 512);
        for (uint2 res : getLinearResolutionLevels(1080, 8)) {
            app->resizeFrameBuffer(res.x, res.y);
            results.push_back({
                {"resolution", {res.x, res.y}},
                {"MP", res.x * res.y / 1e6},
                {"runs", benchmarkPerformance(app, {
                    graphSPPM(false, 0.0f, true, false), // QuerySearch
                    // graphSPPM(true, 0.0f, true, false),  // PhotonSearch // NOTE: This crashes on Blackwell
                    graphSPPM(false, 0.0f, true, true), // StochQuerySearch
                    // graphSPPM(true, 0.0f, true, true),  // StochPhotonSearch // NOTE: This crashes on Blackwell
                    graphSPPM(false, 0.7f, true, false), // QuerySearch
                    graphSPPM(true, 0.7f, true, false),  // PhotonSearch
                    graphSPPM(false, 0.7f, true, true), // StochQuerySearch
                    graphSPPM(true, 0.7f, true, true),  // StochPhotonSearch
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
                    graphPT(), // PT
                    graphNRCSPPC(), // NRC+SPPC
                    graphSPPM(false, 0.7f, true, true), // StochQuerySearch
                }, 32)},
            });
        }
        writeJson(results, resultsDir / "nrc_vs_res.json");
    }

    if (args::get(phnrcVsR)) {
        json results = {};
        auto app = createApp("cornell_box.pyscene", 512);
        for (float r : {0.001f, 0.005f, 0.01f, 0.015f, 0.02f, 0.03f}) {
            logInfo("Running NRC+SPPC vs radius benchmark for r={}", r);
            results.push_back({
                {"r", r},
                {"runs", benchmarkPerformance(app, {
                    graphNRCSPPCSameR(0.0f, true, false, r),
                    graphNRCSPPCSameR(0.0f, true, true, r),
                    graphNRCSPPCSameR(0.7f, true, false, r),
                    graphNRCSPPCSameR(0.7f, true, true, r),
                }, 32)},
            });
        }
        writeJson(results, resultsDir / "phnrc_vs_r.json");
    }

    if (args::get(nrcVariants)) {
        auto app = createApp("cornell_box_caustic.pyscene", 512);
        auto json = benchmarkPerformance(app, {
            graphNRCPT(1), // NRC
            graphNRCPT(32), // NRC+LT
            graphNRCLT(), // NRC+LT
            graphNRCSPPC(), // NRC+SPPC
        }, 128);
        writeJson(json, resultsDir / "nrc_variants.json");
    }

    if (args::get(quality)) {
        for (const auto& scene : {
            "cornell_box_caustic.pyscene",
            "cornell_box.pyscene",
            "cornell_box_bunny.pyscene",
        }) {
            auto app = createApp(scene, 512);
            std::vector<GraphConfigurator> configs = {
                graphNRCPT(1), // NRC
                graphNRCPT(32), // NRC+LT
                graphNRCLT(), // NRC+LT
                graphNRCSPPC(), // NRC+SPPC
                graphSPPM(), // SPPM
            };
            for (const auto& config : configs) {
                auto g = config(app->createRenderGraph());
                benchmarkQuality(app, g, 10.0, getResultsDir("quality"));
            }
        }
    }

    if (args::get(convergenceTest)) {
        const auto& scene = "cornell_box.pyscene";
        auto app = createApp(scene, 512);
        auto ref = ensureReference(app);
        
        logInfo("Running convergence test for scene: {}", scene);
        std::vector<GraphConfigurator> graphs = {
            graphNRCPT(1), // NRC
            graphNRCLT(), // NRC+LT
            graphNRCSPPC(), // NRC+SPPC
        };
        
        auto convergenceResults = benchmarkConvergence(app, graphs, ref, 1.0);
        writeJson(convergenceResults, resultsDir / "convergence.json");
    }

    if (args::get(ltTest)) {
        auto app = createApp("veach-ajar/scene-v4.pbrt", 512);
        for (uint mode = 0; mode < 3; ++mode) {
            auto g = graphNRCLT(10, false, mode)(app->createRenderGraph());
            benchmarkQuality(app, g, 10.0, getResultsDir("lt"));
        }
    }

    if (args::get(sppmTest)) {
        auto app = createApp("cornell_box_caustic.pyscene", 512);
        std::vector<GraphConfigurator> configs = {
            rename(graphSPPM(false, 0.7f, true, false), "QuerySearch"),
            rename(graphSPPM(true, 0.7f, true, false), "PhotonSearch"),
            rename(graphSPPM(false, 0.7f, true, true), "StochQuerySearch"),
            rename(graphSPPM(true, 0.7f, true, true), "StochPhotonSearch"),
        };
        for (const auto& config : configs) {
            auto g = config(app->createRenderGraph());
            benchmarkQuality(app, g, 10.0, getResultsDir("sppm"));
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