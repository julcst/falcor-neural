add_plugin(DebugQueryBuffer)

target_sources(DebugQueryBuffer PRIVATE
    DebugQueryBuffer.cpp
    DebugQueryBuffer.h
    DebugQueryBuffer.cs.slang
)

target_copy_shaders(DebugQueryBuffer RenderPasses/DebugQueryBuffer)

target_source_group(DebugQueryBuffer "RenderPasses")
