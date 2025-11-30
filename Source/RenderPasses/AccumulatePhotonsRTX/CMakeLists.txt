add_plugin(AccumulatePhotonsRTX)

target_sources(AccumulatePhotonsRTX PRIVATE
    AccumulatePhotonsRTX.cpp
    AccumulatePhotonsRTX.h
    AccumulatePhotons.rt.slang
    Preparation.cs.slang
    Structs.slang
    Finalize.cs.slang
)

target_copy_shaders(AccumulatePhotonsRTX RenderPasses/AccumulatePhotonsRTX)

target_source_group(AccumulatePhotonsRTX "RenderPasses")
