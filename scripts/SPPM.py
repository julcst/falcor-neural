try:
    from falcor import *  # type: ignore
except ImportError:
    # Allow static analysis or running outside Falcor's Python host.
    pass

# Fallback stubs for static analyzers when running outside Falcor
if "RenderGraph" not in globals():
    class RenderGraph:  # type: ignore
        def __init__(self, name):
            pass
        def addPass(self, *args, **kwargs):
            pass
        def addEdge(self, *args, **kwargs):
            pass
        def markOutput(self, *args, **kwargs):
            pass
if "createPass" not in globals():
    def createPass(name, opts=None):  # type: ignore
        return object()
if "m" not in globals():
    m = None  # type: ignore


def render_graph_SPPM():
    g = RenderGraph("SPPM")

    sppm = createPass("SPPM")
    g.addPass(sppm, "SPPM")

    accum = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(accum, "AccumulatePass")

    tonemap = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(tonemap, "ToneMapper")

    # Wire: SPPM.color -> AccumulatePass.input -> ToneMapper.src
    g.addEdge("SPPM.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    g.markOutput("ToneMapper.dst")
    return g


SPPM = render_graph_SPPM()
if m is not None:
    m.addGraph(SPPM)