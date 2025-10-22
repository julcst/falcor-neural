from falcor import *

def render_graph_SPPM():
    g = RenderGraph("SPPM")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    SPPM = createPass("SPPM")
    g.addPass(SPPM, "SPPM")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

SPPM = render_graph_SPPM()
try: m.addGraph(SPPM)
except NameError: None