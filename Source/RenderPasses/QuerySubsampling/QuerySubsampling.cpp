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
#include "QuerySubsampling.h"
#include "../TraceQueries/Query.slang"
#include "../NRC/NRC.slang"

namespace
{
const char kShaderFile[] = "RenderPasses/QuerySubsampling/QuerySubsampling.cs.slang";

const char kInputQueries[] = "queries";
const char kOutputQueries[] = "sample";
const char kInputNRC[] = "nrcInput";
const char kOutputNRC[] = "nrcOutput";

const char kOutputCount[] = "count";
const char kReplacementFactor[] = "replacementFactor";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, QuerySubsampling>();
}

QuerySubsampling::QuerySubsampling(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    setProperties(props);
}

void QuerySubsampling::setProperties(const Properties& props)
{
    for (const auto& [key, value] : props) {
        if (key == kOutputCount) mOutputCount = value;
        else if (key == kReplacementFactor) mReplacementFactor = value;
        else logWarning("Unrecognized property '{}' in QuerySubsampling render pass.", key);
    }
}

Properties QuerySubsampling::getProperties() const
{
    Properties props;
    props[kOutputCount] = mOutputCount;
    props[kReplacementFactor] = mReplacementFactor;
    return props;
}

void QuerySubsampling::renderUI(Gui::Widgets& widget)
{
    if (widget.var("Output Count", mOutputCount, NRC_BATCH_SIZE_GRANULARITY, 1u << 20u, NRC_BATCH_SIZE_GRANULARITY))
    {
        requestRecompile();
    }
    widget.var("Replacement Factor", mReplacementFactor, 0.0f, 1.0f);
}

RenderPassReflection QuerySubsampling::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputQueries, "Buffer containing input queries.")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    reflector.addOutput(kOutputQueries, "Buffer containing randomly subsampled queries.")
        .rawBuffer(mOutputCount * sizeof(Query))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);

    reflector.addInput(kInputNRC, "Buffer containing input NRC inputs.")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .flags(RenderPassReflection::Field::Flags::Optional);
    reflector.addOutput(kOutputNRC, "Buffer containing randomly subsampled NRC inputs.")
        .rawBuffer(mOutputCount * sizeof(NRCInput))
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource)
        .flags(RenderPassReflection::Field::Flags::Optional);

    return reflector;
}

void QuerySubsampling::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "QuerySubsampling");

    const auto pInputQueries = renderData[kInputQueries]->asBuffer();
    const auto pOutputQueries = renderData[kOutputQueries]->asBuffer();
    
    FALCOR_ASSERT(pInputQueries);
    FALCOR_ASSERT(pOutputQueries);

    const uint32_t inputCount = pInputQueries->getSize() / sizeof(Query);
    
    if (inputCount == 0)
    {
        logWarning("Input query buffer is empty.");
        return;
    }
    
    auto pInputNRC = renderData[kInputNRC];
    auto pOutputNRC = renderData[kOutputNRC];

    bool nrcInputs = pInputNRC && pOutputNRC;

    preparePass();
    mpSubsamplePass->addDefine("SUBSAMPLE_NRC", nrcInputs ? "1" : "0");

    const uint32_t replaceCount = static_cast<uint32_t>(mOutputCount * mReplacementFactor);
    const uint32_t replaceBegin = mLastReplaced;
    mLastReplaced = (mLastReplaced + replaceCount) % mOutputCount;

    auto var = mpSubsamplePass->getRootVar();
    var["gInputQueries"] = pInputQueries;
    var["gOutputQueries"] = pOutputQueries;
    var["CB"]["gInputCount"] = inputCount;
    var["CB"]["gOutputCount"] = mOutputCount;
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gReplaceBegin"] = replaceBegin;
    var["CB"]["gReplaceCount"] = replaceCount;
    if (nrcInputs) {
        var["gInputNRC"] = pInputNRC->asBuffer();
        var["gOutputNRC"] = pOutputNRC->asBuffer();
    }

    logInfo("Replacing {}/{}, sampled from {} queries {} NRC", replaceCount, mOutputCount, inputCount, nrcInputs ? "with" : "without");
    mpSubsamplePass->execute(pRenderContext, replaceCount, 1);

    mFrameCount++;
}

void QuerySubsampling::preparePass()
{
    if (!mpSubsamplePass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kShaderFile);
        desc.csEntry("main");

        mpSubsamplePass = ComputePass::create(mpDevice, desc);
    }
}
