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
#include "PhotonSampler.h"
#include "Scene/Scene.h"
#include "Core/Program/Program.h"

namespace Falcor
{
    PhotonSampler::PhotonSampler(RenderContext* pRenderContext, ref<Scene> pScene)
        : mpScene(pScene)
    {
        if (mpScene->useEnvLight())
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(pRenderContext->getDevice(), mpScene->getEnvMap());
        }

        // NOTE: Important to get light collection before checking useEmissiveLights()
        if (auto pLights = mpScene->getILightCollection(pRenderContext))
        {
            if (mpScene->useEmissiveLights())
            {
                mpEmissiveSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, pLights);
            }
        }
    }

    ref<PhotonSampler> PhotonSampler::create(RenderContext* pRenderContext, ref<Scene> pScene)
    {
        return make_ref<PhotonSampler>(pRenderContext, pScene);
    }

    bool PhotonSampler::update(RenderContext* pRenderContext)
    {
        bool updated = false;

        // Update EnvMapSampler
        if (mpScene->useEnvLight())
        {
            if (!mpEnvMapSampler)
            {
                mpEnvMapSampler = std::make_unique<EnvMapSampler>(pRenderContext->getDevice(), mpScene->getEnvMap());
                updated = true;
            }
            // EnvMapSampler doesn't have an update method, it just holds the map.
            // But if the map changes in the scene, we might need to recreate it?
            // For now assuming static env map or handled by scene.
        }
        else
        {
            mpEnvMapSampler.reset();
        }

        // Update EmissivePowerSampler
        // NOTE: Important to get light collection before checking useEmissiveLights()
        auto pLights = mpScene->getILightCollection(pRenderContext);
        if (pLights && mpScene->useEmissiveLights())
        {
            pLights->prepareSyncCPUData(pRenderContext);
            if (!mpEmissiveSampler)
            {
                mpEmissiveSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, pLights);
                updated = true;
            }
            else
            {
                updated |= mpEmissiveSampler->update(pRenderContext, pLights);
            }
        }
        else
        {
            mpEmissiveSampler.reset();
        }

        return updated;
    }

    void PhotonSampler::bindShaderData(const ShaderVar& var) const
    {
        if (mpEnvMapSampler)
        {
            mpEnvMapSampler->bindShaderData(var["envMapSampler"]);
        }
        if (mpEmissiveSampler)
        {
            mpEmissiveSampler->bindShaderData(var["emissiveSampler"]);
        }
    }

    DefineList PhotonSampler::getDefines() const
    {
        DefineList defines;
        defines.add("USE_ENV_LIGHT", (mpScene->useEnvLight() && mpEnvMapSampler) ? "1" : "0");
        defines.add("USE_EMISSIVE_LIGHTS", (mpScene->useEmissiveLights() && mpEmissiveSampler) ? "1" : "0");
        defines.add("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");

        if (mpEmissiveSampler)
        {
            defines.add(mpEmissiveSampler->getDefines());
        }
        return defines;
    }
}
