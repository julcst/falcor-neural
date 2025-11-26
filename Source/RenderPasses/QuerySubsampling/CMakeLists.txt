add_plugin(QuerySubsampling)

target_sources(QuerySubsampling PRIVATE
    QuerySubsampling.cpp
    QuerySubsampling.h
)

target_source_group(QuerySubsampling "RenderPasses")
