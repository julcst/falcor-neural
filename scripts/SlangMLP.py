from falcor import *


def render_graph_SlangMLP():
    g = RenderGraph("SlangMLP")

    image = createPass(
        "ImageLoader",
        {
            "filename": "media/test_images/monalisa.jpg",
            "outputFormat": "RGBA32Float",
        },
    )
    g.addPass(image, "ImageLoader")

    mlp = createPass(
        "SlangMLP",
        {
            "trainSteps": 512,
            "learningRate": 5e-2,
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
