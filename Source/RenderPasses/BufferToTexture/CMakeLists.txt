add_plugin(BufferToTexture)

target_sources(BufferToTexture PRIVATE
    BufferToTexture.cpp
    BufferToTexture.h
    BufferToTexture.cs.slang
)

target_copy_shaders(BufferToTexture RenderPasses/BufferToTexture)

target_source_group(BufferToTexture "RenderPasses")
