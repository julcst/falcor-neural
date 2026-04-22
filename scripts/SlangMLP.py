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
