add_plugin(QuerySubsampling)

target_sources(QuerySubsampling PRIVATE
    QuerySubsampling.cpp
    QuerySubsampling.h
    QuerySubsampling.cs.slang
)

target_copy_shaders(QuerySubsampling RenderPasses/QuerySubsampling)

target_source_group(QuerySubsampling "RenderPasses")
