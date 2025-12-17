add_falcor_executable(Testbed)

target_sources(Testbed PRIVATE
    Testbed.cpp
)

target_link_libraries(Testbed PRIVATE Falcor)

# Rebuild after any plugins change.
get_property(ALL_FALCOR_PLUGINS GLOBAL PROPERTY FALCOR_PLUGIN_TARGETS)

if(ALL_FALCOR_PLUGINS)
    add_dependencies(Testbed ${ALL_FALCOR_PLUGINS})
endif()

target_source_group(Testbed "Tools")