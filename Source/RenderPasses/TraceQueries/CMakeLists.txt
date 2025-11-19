add_plugin(TraceQueries)

target_sources(TraceQueries PRIVATE
    Query.slang
    TraceQueries.cpp
    TraceQueries.h
    TraceQueries.rt.slang
)

target_copy_shaders(TraceQueries RenderPasses/TraceQueries)

target_source_group(TraceQueries "RenderPasses")
