add_falcor_executable(PhotonNRCPaper)

target_sources(PhotonNRCPaper PRIVATE
    PhotonNRCPaper.cpp
)

target_link_libraries(PhotonNRCPaper PRIVATE Falcor args)

# Rebuild after any plugins change.
get_property(ALL_FALCOR_PLUGINS GLOBAL PROPERTY FALCOR_PLUGIN_TARGETS)

if(ALL_FALCOR_PLUGINS)
    add_dependencies(PhotonNRCPaper ${ALL_FALCOR_PLUGINS})
endif()

target_source_group(PhotonNRCPaper "Tools")