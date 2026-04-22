add_plugin(SlangMLP)

target_sources(SlangMLP PRIVATE
    SlangMLP.cpp
    SlangMLP.h
    Training.cs.slang
    Optimization.cs.slang
    Inference.cs.slang
    Config.slang
)

target_copy_shaders(SlangMLP RenderPasses/SlangMLP)

target_source_group(SlangMLP "RenderPasses")
