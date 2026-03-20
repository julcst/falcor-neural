from falcor import *


def render_graph_PT():
    g = RenderGraph("PT")
    g.createPass("accum", "AccumulatePass", {"precisionMode": "Double"})
    g.createPass("pt", "PathTracer", {"maxDiffuseBounces": 8, "maxSpecularBounces": 8})
    g.createPass("vbuff", "VBufferRT", {})

    g.addEdge("vbuff.vbuffer", "pt.vbuffer")
    g.addEdge("vbuff.viewW", "pt.viewW")
    g.addEdge("vbuff.mvec", "pt.mvec")
    g.addEdge("pt.color", "accum.input")
    g.markOutput("accum.output")
    return g


def render_graph_SPPM(reverseSearch=False, rejProb=0.0, rr=True, stoch=True, reduction=True):
    g = RenderGraph("SPPM")
    g.createPass("VisualizePhotons", "VisualizePhotons", {})
    g.createPass(
        "TracePhotons",
        "TracePhotons",
        {
            "photonCount": 1 << 20,
            "maxBounces": 8,
            "globalRejectionProb": rejProb,
            "useRussianRoulette": rr,
        },
    )

    accum_props = {
        "visualizeHeatmap": False,
        "globalRadius": 0.015,
        "causticRadius": 0.003,
        "stochEval": stoch,
        "reverseSearch": reverseSearch,
    }
    if not reduction:
        accum_props["gloablAlpha"] = 1.0
        accum_props["causticAlpha"] = 1.0

    g.createPass("AccumPh", "AccumulatePhotonsRTX", accum_props)
    g.createPass("TraceQueries", "TraceQueries", {"resetStatisticsPerFrame": False})

    g.addEdge("TracePhotons.photons", "VisualizePhotons.photons")
    g.addEdge("TracePhotons.counters", "VisualizePhotons.counters")

    g.addEdge("TracePhotons.photons", "AccumPh.photons")
    g.addEdge("TracePhotons.counters", "AccumPh.photonCounters")
    g.addEdge("TraceQueries.queries", "AccumPh.queries")

    g.markOutput("AccumPh.outputTexture")
    return g


def render_graph_NRCSPPC(
    rej=0.7,
    stoch=True,
    globalR=0.015,
    causticR=0.003,
    photonCount=1 << 19,
    replacement=0.02,
    debug=True,
):
    g = RenderGraph("NRC+SPPC+Debug" if debug else "NRC+SPPC")
    g.createPass(
        "TracePhotons",
        "TracePhotons",
        {"photonCount": photonCount, "maxBounces": 8, "globalRejectionProb": rej},
    )
    g.createPass(
        "AccumPh",
        "AccumulatePhotonsRTX",
        {
            "visualizeHeatmap": False,
            "globalRadius": globalR,
            "causticRadius": causticR,
            "stochEval": stoch,
        },
    )
    g.createPass("Accum", "AccumulatePass", {})
    g.createPass("TraceQueries", "TraceQueries", {"resetStatisticsPerFrame": True})
    g.createPass("qsamp", "QuerySubsampling", {"replacementFactor": replacement})
    g.createPass("nrc", "NRC", {"jitFusion": True})

    g.addEdge("TraceQueries.queries", "qsamp.queries")
    g.addEdge("TraceQueries.nrcInput", "qsamp.nrcInput")

    g.addEdge("TracePhotons.photons", "AccumPh.photons")
    g.addEdge("TracePhotons.counters", "AccumPh.photonCounters")
    g.addEdge("qsamp.sample", "AccumPh.queries")

    g.addEdge("qsamp.nrcOutput", "nrc.trainInput")
    g.addEdge("AccumPh.outputBuffer", "nrc.trainTarget")
    g.addEdge("TraceQueries.nrcInput", "nrc.inferenceInput")
    g.addEdge("TraceQueries.queries", "nrc.inferenceQueries")

    g.markOutput("nrc.output")

    if debug:
        g.createPass("visPh", "VisualizePhotons", {})
        g.addEdge("TracePhotons.photons", "visPh.photons")
        g.addEdge("TracePhotons.counters", "visPh.counters")
        g.markOutput("visPh.dst")

        g.createPass("visQueries", "VisualizeQueries", {})
        g.addEdge("qsamp.sample", "visQueries.queries")
        g.addEdge("AccumPh.queryStates", "visQueries.queryStates")
        g.markOutput("visQueries.output")

    return g


def render_graph_NRCPT(spp=1):
    g = RenderGraph("NRC+PT" if spp == 1 else f"NRC+PT{spp}")
    g.createPass("TraceQueries", "TraceQueries", {})
    g.createPass("qsamp", "QuerySubsampling", {})
    g.createPass("nrc", "NRC", {"jitFusion": True})
    g.createPass(
        "PTQuery",
        "PathTracerQuery",
        {
            "maxDiffuseBounces": 8,
            "maxSpecularBounces": 8,
            "samplesPerPixel": spp,
            "parallelMultiSampling": True,
        },
    )

    g.addEdge("TraceQueries.queries", "qsamp.queries")
    g.addEdge("TraceQueries.nrcInput", "qsamp.nrcInput")

    g.addEdge("qsamp.sample", "PTQuery.queries")

    g.addEdge("qsamp.nrcOutput", "nrc.trainInput")
    g.addEdge("PTQuery.radiance", "nrc.trainTarget")
    g.addEdge("TraceQueries.nrcInput", "nrc.inferenceInput")
    g.addEdge("TraceQueries.queries", "nrc.inferenceQueries")

    g.markOutput("nrc.output")
    return g


def render_graph_NRCLT(maxBounces=6, visualizeQueries=False, mode=0):
    if mode == 0:
        name = "NRC+LT"
    elif mode == 1:
        name = "NRC+LT+Warp"
    else:
        name = "NRC+LT+Reservoir"

    g = RenderGraph(name)
    g.createPass("TraceQueries", "TraceQueries", {})
    g.createPass("qsamp", "QuerySubsampling", {})
    g.createPass("nrc", "NRC", {"jitFusion": True, "useFactorization": True})
    g.createPass("estim", "PhotonNEE", {"maxBounces": maxBounces, "mode": mode})

    g.addEdge("TraceQueries.queries", "qsamp.queries")
    g.addEdge("TraceQueries.nrcInput", "qsamp.nrcInput")

    g.addEdge("qsamp.sample", "estim.queries")

    g.addEdge("qsamp.nrcOutput", "nrc.trainInput")
    g.addEdge("estim.output", "nrc.trainTarget")
    g.addEdge("TraceQueries.nrcInput", "nrc.inferenceInput")
    g.addEdge("TraceQueries.queries", "nrc.inferenceQueries")

    if visualizeQueries:
        g.createPass("visQueries", "VisualizeQueries", {"constantRadius": 0.01, "mode": 3})
        g.addEdge("qsamp.sample", "visQueries.queries")
        g.addEdge("estim.output", "visQueries.colors")
        g.markOutput("visQueries.output")

    g.markOutput("nrc.output")
    return g


def render_graph_PTQuery(spp=1):
    g = RenderGraph("PT")
    g.createPass("TraceQueries", "TraceQueries", {})
    g.createPass(
        "PTQuery",
        "PathTracerQuery",
        {
            "maxDiffuseBounces": 8,
            "maxSpecularBounces": 8,
            "samplesPerPixel": spp,
            "parallelMultiSampling": True,
        },
    )
    g.createPass("B2T", "BufferToTexture", {})

    g.addEdge("TraceQueries.queries", "PTQuery.queries")
    g.addEdge("PTQuery.radiance", "B2T.input")

    g.markOutput("B2T.output")
    return g


def render_graph_Tonemap():
    g = RenderGraph("Tonemap")
    g.createPass("tonemap", "ToneMapper", {})
    g.markOutput("tonemap.dst")
    return g


def render_graph_FLIP(refPath):
    g = RenderGraph("FLIP")
    g.createPass("ref", "ImageLoader", {"filename": refPath, "outputFormat": "RGBA32Float"})
    g.createPass("flip", "FLIPPass", {"isHDR": True, "computePooledFLIPValues": True})
    g.createPass("error", "ErrorMeasurePass", {"SelectedOutputId": "Difference"})
    g.createPass("tonemap", "ToneMapper", {})

    g.addEdge("ref.dst", "flip.referenceImage")
    g.addEdge("ref.dst", "error.Reference")
    g.markOutput("tonemap.dst")
    g.markOutput("flip.errorMapDisplay")
    g.markOutput("error.Output")
    return g


def _register(g):
    try:
        m.addGraph(g)
    except NameError:
        pass

m.loadScene('../scenes/cornell_box_caustic_animated.pyscene')
_register(render_graph_NRCSPPC(debug=True, replacement = 0.2))
_register(render_graph_NRCSPPC(debug=False, replacement = 0.2))
_register(render_graph_PT())
_register(render_graph_SPPM(reverseSearch=False))
_register(render_graph_NRCPT(1))
_register(render_graph_NRCPT(32))
_register(render_graph_NRCLT(mode=0))
_register(render_graph_PTQuery(1))
_register(render_graph_Tonemap())

# m.frameCapture.addFrames(m.activeGraph, range(0, 1000, 30))