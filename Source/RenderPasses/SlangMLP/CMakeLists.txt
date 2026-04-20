add_plugin(SlangMLP)

target_sources(SlangMLP PRIVATE
    SlangMLP.cpp
    SlangMLP.h
    SlangMLP.cs.slang
)

target_copy_shaders(SlangMLP RenderPasses/SlangMLP)

target_source_group(SlangMLP "RenderPasses")
