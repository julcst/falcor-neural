from falcor import *


def render_graph_SlangMLP():
    g = RenderGraph("SlangMLP")

    image = createPass(
        "ImageLoader",
        {
            "filename": "test_images/monalisa.jpg",
            "outputFormat": "RGBA32Float",
        },
    )
    g.addPass(image, "ImageLoader")

    mlp = createPass(
        "SlangMLP",
        {
            "batchSize": 2048,
            "trainSteps": 1,
            "learningRate": 1e-3,
        },
    )
    g.addPass(mlp, "SlangMLP")

    g.addEdge("ImageLoader.dst", "SlangMLP.src")

    g.markOutput("SlangMLP.dst")
    g.markOutput("ImageLoader.dst")
    return g


SlangMLPGraph = render_graph_SlangMLP()
try:
    m.addGraph(SlangMLPGraph)
except NameError:
    None

# m.clock.exitFrame = 1001
# m.frameCapture.baseFilename = "Mogwai"
# m.frameCapture.addFrames(m.activeGraph, [20, 50, 100, 400, 700, 1000])

# m.profiler.enabled = True
# m.profiler.start_capture()
# for frame in range(256):
#     m.renderFrame()
# capture = m.profiler.end_capture()
# m.profiler.enabled = False

# print(f"Mean frame time: {capture['events']['/onFrameRender/RenderGraphExe::execute()/SlangMLP/SlangMLP/gpu_time']['stats']['mean']} ms")
