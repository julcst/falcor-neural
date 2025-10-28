add_plugin(SPPM)

target_sources(SPPM PRIVATE
    SPPM.cpp
    SPPM.h
    SPPMDebug.slangh
    TracePhotons.rt.slang
    TraceQueries.rt.slang
    IntersectPhotons.rt.slang
    BuildQueryBounds.cs.slang
    BuildQueryGrid.cs.slang
    AccumulateByGrid.cs.slang
    SPPMResolve.ps.slang
    PhotonQuery.slang
    PhotonHit.slang
)

target_copy_shaders(SPPM RenderPasses/SPPM)

target_source_group(SPPM "RenderPasses")
