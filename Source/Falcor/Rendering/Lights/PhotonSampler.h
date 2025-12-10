/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Scene/Lights/LightCollection.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Core/Program/Program.h"
#include "Core/Program/DefineList.h"
#include <memory>

namespace Falcor
{
    class RenderContext;
    struct ShaderVar;

    /** Helper class to sample lights for photon mapping.
        This class aggregates different light samplers (EnvMap, Emissive, Analytic)
        and provides a unified interface for the shader.
    */
    class FALCOR_API PhotonSampler : public Object
    {
        FALCOR_OBJECT(PhotonSampler)
    public:
        /** Creates a PhotonSampler for a given scene.
            \param[in] pRenderContext The render context.
            \param[in] pScene The scene.
        */
        static ref<PhotonSampler> create(RenderContext* pRenderContext, ref<Scene> pScene);

        /** Updates the sampler to the current frame.
            \param[in] pRenderContext The render context.
            \return True if the sampler was updated.
        */
        bool update(RenderContext* pRenderContext);

        /** Bind the light sampler data to a given shader variable.
            \param[in] var Shader variable.
        */
        void bindShaderData(const ShaderVar& var) const;

        /** Get the defines needed for the shader.
            \return List of defines.
        */
        DefineList getDefines() const;

    public:
        PhotonSampler(RenderContext* pRenderContext, ref<Scene> pScene);

    private:
        ref<Scene> mpScene;
        std::unique_ptr<EmissivePowerSampler> mpEmissiveSampler;
        std::unique_ptr<EnvMapSampler> mpEnvMapSampler;
    };
}
