add_plugin(TracePhotons)

target_sources(TracePhotons PRIVATE
    Common.slang
    Structs.slang
    TracePhotons.cpp
    TracePhotons.h
    TracePhotons.rt.slang
)

target_copy_shaders(TracePhotons RenderPasses/TracePhotons)

target_source_group(TracePhotons "RenderPasses")
