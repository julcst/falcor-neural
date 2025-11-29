set(CMAKE_CUDA_ARCHITECTURES 86)
# Force tiny-cuda-nn to generate Position Independent Code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set (target NRC)

# add_plugin with added tiny-cuda-nn
add_library(${target} SHARED)

target_link_libraries(${target} PRIVATE Falcor tiny-cuda-nn)

set_target_properties(${target} PROPERTIES
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY ${FALCOR_PLUGIN_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${FALCOR_PLUGIN_OUTPUT_DIRECTORY}
    INSTALL_RPATH "\$ORIGIN/../"
)

# Add target to global plugin target list.
set_property(GLOBAL APPEND PROPERTY FALCOR_PLUGIN_TARGETS ${target})

target_sources(NRC PRIVATE
    NRC.cpp
    NRC.h
    TCNNBridge.cu
    TCNNBridge.h
)

set_target_properties(NRC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_source_group(NRC "RenderPasses")
