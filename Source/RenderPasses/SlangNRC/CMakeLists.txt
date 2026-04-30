add_plugin(SlangNRC)

target_sources(SlangNRC PRIVATE
    SlangNRC.cpp
    SlangNRC.h
    NRCCompat.slang
    Training.cs.slang
    Optimization.cs.slang
    Inference.cs.slang
    Encodings.slang
    Config.slang
)

target_copy_shaders(SlangNRC RenderPasses/SlangNRC)

target_source_group(SlangNRC "RenderPasses")
