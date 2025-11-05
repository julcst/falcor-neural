from falcor import *

def render_graph_SPPM():
    """Create render graph for SPPM with accumulation and tonemapping."""
    g = RenderGraph("SPPM")
    
    # SPPM pass
    sppm = createPass("SPPM", {})
    g.addPass(sppm, "SPPM")
    
    # Accumulation for progressive rendering
    accum = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(accum, "AccumulatePass")
    
    # Tonemapping for display
    tonemap = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(tonemap, "ToneMapper")
    
    # Connect passes
    g.addEdge("SPPM.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    
    g.markOutput("ToneMapper.dst")
    return g

# Load Cornell Box scene as default
m.loadScene('test_scenes/cornell_box.pyscene')

# Attach render graph
m.addGraph(render_graph_SPPM())