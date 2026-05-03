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


def render_graph_NRCPT(spp=1):
    g = RenderGraph("NRC+PT" if spp == 1 else f"NRC+PT{spp}")
    g.createPass("TraceQueries", "TraceQueries", {})
    g.createPass("qsamp", "QuerySubsampling", {})
    g.createPass("nrc", "SlangNRC", {})
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


def render_graph_CudaNRCPT(spp=1):
    g = RenderGraph("CudaNRC+PT" if spp == 1 else f"CudaNRC+PT{spp}")
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


m.loadScene("../scenes/cornell_box_bunny.pyscene")
m.addGraph(render_graph_NRCPT(32))
m.addGraph(render_graph_CudaNRCPT(32))
m.addGraph(render_graph_PT())

for frame in range(4):
    m.renderFrame()

m.timingCapture.captureFrameTime("timings.csv")
m.profiler.enabled = True
m.profiler.start_capture()
for frame in range(32):
    m.renderFrame()
capture = m.profiler.end_capture()

print(f"Mean frame time: {capture['events']['/onFrameRender/gpu_time']['stats']['mean']} ms")

exit()
