add_plugin(AccumulatePhotonsRTX)

target_sources(AccumulatePhotonsRTX PRIVATE
    AccumulatePhotonsRTX.cpp
    AccumulatePhotonsRTX.h
    QuerySearch.rt.slang
    PhotonSearch.rt.slang
    Preparation.cs.slang
    PreparePhotons.cs.slang
    Structs.slang
    FinalizeTexture.cs.slang
    FinalizeBuffer.cs.slang
)

target_copy_shaders(AccumulatePhotonsRTX RenderPasses/AccumulatePhotonsRTX)

target_source_group(AccumulatePhotonsRTX "RenderPasses")
