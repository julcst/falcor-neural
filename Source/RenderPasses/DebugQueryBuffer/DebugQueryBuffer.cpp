/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "DebugQueryBuffer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "../TraceQueries/Query.slang"
#include "../NRC/NRC.slang"

namespace
{
const char kShaderFile[] = "RenderPasses/DebugQueryBuffer/DebugQueryBuffer.cs.slang";

const char kInputQueries[] = "queries";
const char kInputNRC[] = "nrcInput";

const ChannelList kOutputChannels = {
    // Query fields
    {"queryPosition", "gQueryPosition", "Query position (xyz)", true, ResourceFormat::RGBA32Float},
    {"queryViewDir", "gQueryViewDir", "Query view direction (xyz)", true, ResourceFormat::RGBA32Float},
    {"queryEmission", "gQueryEmission", "Query emission (rgb)", true, ResourceFormat::RGBA32Float},
    {"queryNormal", "gQueryNormal", "Query normal (xyz)", true, ResourceFormat::RGBA32Float},
    {"queryThroughput", "gQueryThroughput", "Query throughput (rgb)", true, ResourceFormat::RGBA32Float},
    {"queryGeomID", "gQueryGeomID", "Query geometry ID", true, ResourceFormat::R32Uint},
    // NRCInput fields
    {"nrcPosition", "gNRCPosition", "NRC position (xyz)", true, ResourceFormat::RGBA32Float},
    {"nrcRoughness", "gNRCRoughness", "NRC roughness (r)", true, ResourceFormat::R32Float},
    {"nrcWo", "gNRCWo", "NRC outgoing direction (xyz)", true, ResourceFormat::RGBA32Float},
    {"nrcWn", "gNRCWn", "NRC normal (xyz)", true, ResourceFormat::RGBA32Float},
    {"nrcDiffuse", "gNRCDiffuse", "NRC diffuse (rgb)", true, ResourceFormat::RGBA32Float},
    {"nrcSpecular", "gNRCSpecular", "NRC specular (rgb)", true, ResourceFormat::RGBA32Float},
};
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, DebugQueryBuffer>();
}

DebugQueryBuffer::DebugQueryBuffer(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {}

Properties DebugQueryBuffer::getProperties() const
{
    return {};
}

RenderPassReflection DebugQueryBuffer::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    
    reflector.addInput(kInputQueries, "Buffer containing Query structures")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::ShaderResource);
    
    reflector.addInput(kInputNRC, "Buffer containing NRCInput structures")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .flags(RenderPassReflection::Field::Flags::Optional);
    
    addRenderPassOutputs(reflector, kOutputChannels);
    
    return reflector;
}

void DebugQueryBuffer::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "DebugQueryBuffer");

    if (!mpScene) return;
    
    const auto pQueryBuffer = renderData[kInputQueries]->asBuffer();
    if (!pQueryBuffer)
    {
        logWarning("DebugQueryBuffer: No query buffer provided.");
        return;
    }
    
    const uint2 frameDim = renderData.getDefaultTextureDims();
    if (frameDim.x == 0 || frameDim.y == 0)
        return;
    
    preparePass();
    
    for (const auto& define : getValidResourceDefines(kOutputChannels, renderData))
    {
        mpDebugPass->addDefine(define.first, define.second);
    }
    
    auto var = mpDebugPass->getRootVar();
    var["CB"]["gFrameDim"] = frameDim;
    var["gQueries"] = pQueryBuffer;
    
    auto pNRCBuffer = renderData[kInputNRC];
    if (pNRCBuffer)
    {
        var["gNRCInput"] = pNRCBuffer->asBuffer();
    }
    
    for (const auto& channel : kOutputChannels)
    {
        ref<Texture> pTexture = renderData.getTexture(channel.name);
        if (pTexture)
        {
            var[channel.texname] = pTexture;
        }
    }
    
    mpDebugPass->execute(pRenderContext, uint3(frameDim, 1));
}

void DebugQueryBuffer::renderUI(Gui::Widgets& widget) {}

void DebugQueryBuffer::setScene(RenderContext * pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
}

void DebugQueryBuffer::preparePass()
{
    if (!mpDebugPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderFile);
        desc.csEntry("main");
        
        mpDebugPass = ComputePass::create(mpDevice, desc, mpScene->getSceneDefines());
    }
}
