from falcor import *

def render_graph_SPPM():
    """Create render graph for SPPM with accumulation and tonemapping."""
    g = RenderGraph("SPPM")

    # Accumulation for progressive rendering
    g.createPass("AccumulatePass", "AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    
    # Tonemapping for display
    g.createPass("ToneMapper", "ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    
    # TracePhotons
    g.createPass("TracePhotons", "TracePhotons")
    g.createPass("VisualizePhotons", "VisualizePhotons")
    g.addEdge("TracePhotons.photons", "VisualizePhotons.photons")
    g.addEdge("TracePhotons.counters", "VisualizePhotons.counters")
    g.addEdge("VisualizePhotons.dst", "ToneMapper.src")  # Temporary connection for visualization

    # Reference PathTracer
    g.createPass("PathTracer", "PathTracer", {'samplesPerPixel': 1})
    g.createPass("VBufferRT", "VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    
    # Connect passes
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    #g.addEdge("AccumulatePass.output", "ToneMapper.src")
    
    g.markOutput("ToneMapper.dst")
    return g

# Load Cornell Box scene as default
#m.loadScene('test_scenes/cornell_box.pyscene')

# Attach render graph
m.addGraph(render_graph_SPPM())