from falcor import *

def render_graph_SPPM():
    """Create render graph for SPPM with accumulation and tonemapping."""
    g = RenderGraph("SPPM")

    # Accumulation for progressive rendering
    accum = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(accum, "AccumulatePass")
    
    # Tonemapping for display
    tonemap = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(tonemap, "ToneMapper")
    
    # TracePhotons
    photon = createPass("TracePhotons")
    g.addPass(photon, "TracePhotons")
    visualizePhotons = createPass("VisualizePhotons")
    g.addPass(visualizePhotons, "VisualizePhotons")
    g.addEdge("TracePhotons.photons", "VisualizePhotons.photons")
    g.addEdge("TracePhotons.counters", "VisualizePhotons.counters")
    g.addEdge("VisualizePhotons.dst", "ToneMapper.src")  # Temporary connection for visualization

    # Reference PathTracer
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1})
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    
    # Connect passes
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    #g.addEdge("AccumulatePass.output", "ToneMapper.src")
    
    g.markOutput("ToneMapper.dst")
    return g

# Load Cornell Box scene as default
m.loadScene('test_scenes/cornell_box.pyscene')

# Attach render graph
m.addGraph(render_graph_SPPM())