add_plugin(VisualizeQueries)

target_sources(VisualizeQueries PRIVATE
    VisualizeQueries.cpp
    VisualizeQueries.h
    VisualizeQueries.3d.slang
)

target_copy_shaders(VisualizeQueries RenderPasses/VisualizeQueries)

target_source_group(VisualizeQueries "RenderPasses")