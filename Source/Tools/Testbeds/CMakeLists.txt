add_falcor_executable(TestSPPM)

target_sources(TestSPPM PRIVATE
    TestSPPM.cpp
)

target_link_libraries(TestSPPM PRIVATE Falcor)

# Rebuild after any plugins change.
get_property(ALL_FALCOR_PLUGINS GLOBAL PROPERTY FALCOR_PLUGIN_TARGETS)
if(ALL_FALCOR_PLUGINS)
    add_dependencies(TestSPPM ${ALL_FALCOR_PLUGINS})
endif()

target_source_group(TestSPPM "Tools")