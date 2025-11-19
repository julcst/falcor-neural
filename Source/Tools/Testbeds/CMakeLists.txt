add_falcor_executable(TestSPPM)

target_sources(TestSPPM PRIVATE
    TestSPPM.cpp
)

target_link_libraries(TestSPPM PRIVATE Falcor)

target_source_group(TestSPPM "Tools")