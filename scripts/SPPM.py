from falcor import *


def graphPT():
    g = RenderGraph("PT")
    g.createPass("accum", "AccumulatePass", {})
    g.createPass("pt", "PathTracer", {"maxDiffuseBounces": 8, "maxSpecularBounces": 8})
    g.createPass("vbuff", "VBufferRT", {})
    g.addEdge("vbuff.vbuffer", "pt.vbuffer")
    g.addEdge("vbuff.viewW", "pt.viewW")
    g.addEdge("vbuff.mvec", "pt.mvec")
    g.addEdge("pt.color", "accum.input")
    g.markOutput("accum.output")
    return g

m.addGraph(graphPT())


def graphSPPM(reverseSearch=False):
    g = RenderGraph("SPPM_ReverseSearch" if reverseSearch else "SPPM")
    g.createPass("Ref", "ImageLoader", {"filename": "out_ref.exr"})
    g.createPass("VisualizePhotons", "VisualizePhotons", {})
    g.createPass("TracePhotons", "TracePhotons", {"photonCount": 1 << 20})
    g.createPass(
        "AccumPh",
        "AccumulatePhotonsRTX",
        {
            "visualizeHeatmap": False,
            "globalRadius": 0.01,
            "causticRadius": 0.002,
            "reverseSearch": reverseSearch,
        },
    )
    g.createPass("Accum", "AccumulatePass", {})
    g.createPass("TraceQueries", "TraceQueries", {"resetStatisticsPerFrame": False})
    g.createPass("Error", "ErrorMeasurePass", {"SelectedOutputId": "Difference"})
    g.addEdge("TracePhotons.photons", "VisualizePhotons.photons")
    g.addEdge("TracePhotons.counters", "VisualizePhotons.counters")
    g.addEdge("TracePhotons.photons", "AccumPh.photons")
    g.addEdge("TracePhotons.counters", "AccumPh.photonCounters")
    g.addEdge("TraceQueries.queries", "AccumPh.queries")
    g.addEdge("AccumPh.outputTexture", "Error.Source")
    g.addEdge("Ref.dst", "Error.Reference")
    g.markOutput("AccumPh.outputTexture")
    g.markOutput("VisualizePhotons.dst")
    return g

m.addGraph(graphSPPM(reverseSearch=True))
m.addGraph(graphSPPM(reverseSearch=False))


def graphPhotonNRC():
    g = RenderGraph("PhotonNRC")
    g.createPass("TracePhotons", "TracePhotons", {"photonCount": 1 << 22})
    g.createPass(
        "AccumPh",
        "AccumulatePhotonsRTX",
        {"visualizeHeatmap": False, "globalRadius": 0.01, "causticRadius": 0.004},
    )
    g.createPass("Accum", "AccumulatePass", {})
    g.createPass("TraceQueries", "TraceQueries", {"resetStatisticsPerFrame": True})
    g.createPass("qsamp", "QuerySubsampling", {"count": 1 << 14})
    g.createPass("nrc", "NRC", {})
    g.createPass("visPh", "VisualizePhotons", {})
    g.createPass("debug", "DebugQueryBuffer", {})
    g.addEdge("TracePhotons.photons", "visPh.photons")
    g.addEdge("TracePhotons.counters", "visPh.counters")
    g.addEdge("TraceQueries.queries", "qsamp.queries")
    g.addEdge("TraceQueries.nrcInput", "qsamp.nrcInput")
    g.addEdge("TracePhotons.photons", "AccumPh.photons")
    g.addEdge("TracePhotons.counters", "AccumPh.photonCounters")
    g.addEdge("qsamp.sample", "AccumPh.queries")
    g.addEdge("qsamp.nrcOutput", "nrc.trainInput")
    g.addEdge("AccumPh.outputBuffer", "nrc.trainTarget")
    g.addEdge("TraceQueries.nrcInput", "nrc.inferenceInput")
    g.addEdge("TraceQueries.queries", "nrc.inferenceQueries")
    g.addEdge("TraceQueries.queries", "debug.queries")
    g.addEdge("TraceQueries.nrcInput", "debug.nrcInput")
    g.markOutput("nrc.output")
    return g

m.addGraph(graphPhotonNRC())


def graphNRC():
    g = RenderGraph("NRC")
    g.createPass("TraceQueries", "TraceQueries", {})
    g.createPass("qsamp", "QuerySubsampling", {"count": 1 << 16})
    g.createPass("nrc", "NRC", {})
    g.createPass(
        "PTQuery", "PathTracerQuery", {"maxDiffuseBounces": 8, "maxSpecularBounces": 8}
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

m.addGraph(graphNRC())


def graphPTQuery():
    g = RenderGraph("PTQuery")
    g.createPass("TraceQueries", "TraceQueries", {})
    g.createPass("PTQuery", "PathTracerQuery", {})
    g.createPass("B2T", "BufferToTexture", {})
    g.addEdge("TraceQueries.queries", "PTQuery.queries")
    g.addEdge("PTQuery.radiance", "B2T.input")
    g.markOutput("B2T.output")
    return g

m.addGraph(graphPTQuery())


# Load Cornell Box scene as default
# m.loadScene('test_scenes/cornell_box.pyscene')

# Attach render graph
