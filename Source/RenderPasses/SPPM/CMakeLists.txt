add_plugin(SPPM)

target_sources(SPPM PRIVATE
    SPPM.cpp
    SPPM.h
    SPPMDebug.slangh
    SPPMResolve.ps.slang
    TracePhotons.rt.slang
    TraceQueries.rt.slang
    IntersectPhotons.rt.slang
    BuildQueryBounds.cs.slang
    BuildQueryGrid.cs.slang
    AccumulateByGrid.cs.slang
    AccumulateByScreen.cs.slang
    VisualizePhotons.cs.slang
    PhotonQuery.slang
    PhotonHit.slang
)

target_copy_shaders(SPPM RenderPasses/SPPM)

target_source_group(SPPM "RenderPasses")
