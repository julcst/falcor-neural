add_plugin(VisualizePhotons)

target_sources(VisualizePhotons PRIVATE
    VisualizePhotons.cpp
    VisualizePhotons.h
    VisualizePhotons.cs.slang
)

target_copy_shaders(VisualizePhotons RenderPasses/VisualizePhotons)

target_source_group(VisualizePhotons "RenderPasses")
