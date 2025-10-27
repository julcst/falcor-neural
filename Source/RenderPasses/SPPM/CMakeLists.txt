add_plugin(SPPM)

target_sources(SPPM PRIVATE
    SPPM.cpp
    SPPM.h
    TracePhotons.rt.slang
    TraceQueries.rt.slang
    BuildQueryBounds.cs.slang
    ResolveQueries.cs.slang
    ResolveQueries.ps.slang
    PhotonQuery.slang
)

target_copy_shaders(SPPM RenderPasses/SPPM)

target_source_group(SPPM "RenderPasses")
