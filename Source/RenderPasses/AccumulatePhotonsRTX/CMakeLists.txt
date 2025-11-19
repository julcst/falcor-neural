add_plugin(AccumulatePhotonsRTX)

target_sources(AccumulatePhotonsRTX PRIVATE
    AccumulatePhotonsRTX.cpp
    AccumulatePhotonsRTX.h
)

target_source_group(AccumulatePhotonsRTX "RenderPasses")
