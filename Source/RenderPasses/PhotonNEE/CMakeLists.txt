add_plugin(PhotonNEE)

target_sources(PhotonNEE PRIVATE
    PhotonNEE.cpp
    PhotonNEE.h
    PhotonNEE.rt.slang
)

target_copy_shaders(PhotonNEE RenderPasses/PhotonNEE)

target_source_group(PhotonNEE "RenderPasses")
