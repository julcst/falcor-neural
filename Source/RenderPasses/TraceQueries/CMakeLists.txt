add_plugin(TraceQueries)

target_sources(TraceQueries PRIVATE
    TraceQueries.cpp
    TraceQueries.h
)

target_source_group(TraceQueries "RenderPasses")
