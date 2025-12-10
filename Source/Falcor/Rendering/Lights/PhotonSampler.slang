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
/**
 * Photon sampler (derived from RenderPasses/PathTracer/PathTracer.slang and ReSTIR_FG)
 */
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissivePowerSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Utils.Math.MathHelpers;
import Utils.Math.Ray;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Scene;
import Scene.Shading;

#ifndef USE_ENV_LIGHT
#define USE_ENV_LIGHT false
#endif
#ifndef USE_EMISSIVE_LIGHTS
#define USE_EMISSIVE_LIGHTS false
#endif
#ifndef USE_ANALYTIC_LIGHTS
#define USE_ANALYTIC_LIGHTS false
#endif

/** Describes a light sample.
*/
struct PhotonLightSample
{
    float3  flux;
    // float   pdf;
    float3  posW;
    float3  dir;

    Ray getRay() { return Ray(posW, dir, 0.0, FLT_MAX); }
};

struct PhotonSampler {
    // Samplers
    EnvMapSampler envMapSampler;                    ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissivePowerSampler emissiveSampler;           ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    /*******************************************************************
                                Static members
    *******************************************************************/

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;

    /** Types of samplable lights.
    */
    enum class LightSamplerType
    {
        EnvMap,
        Emissive,
        Analytic
    };

    /** Generates a light sample on the environment map.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(inout SampleGenerator sg, out PhotonLightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.flux = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        // ls.pdf = lightSample.pdf;
        ls.posW = -1e30 * lightSample.dir; // Place the origin far away
        ls.dir = lightSample.dir;

        return any(ls.flux > 0.f);
    }

    /** Transforms direction from local to world space
        \param[in] Face Normal
        \param[in] local direction
        \return world space direction
    */
    float3 fromLocalToWorld(const in float3 normal, float3 w)
    {
        // transform from local space -> random dir vector in world space
        float3 bitangent = perp_stark(normal);
        float3 tangent = cross(bitangent, normal);

        return w.x * tangent + w.y * bitangent + w.z * normal;
    }

//#if USE_EMISSIVE_LIGHTS // Function is only valid if the emissive sampler is set
    /** Creates a photon light sample from emissive triangles weighted by emissive power
        \param[in,out] Random Number Generator
        \param[out] Photon light sample
        \return true if light sample is valid
    */
    bool generateEmissiveSample(inout SampleGenerator sg, out PhotonLightSample pls) {
        pls = {};
        if (gScene.lightCollection.isEmpty()) return false;

        // Randomly pick a triangle out of the global list with uniform probability
        const uint triangleCount = gScene.lightCollection.triangleCount;
        // Safety precaution as the result of the multiplication may be rounded to triangleCount even if uLight < 1.0 when triangleCount is large.
        uint triangleIndex = min((uint)(sampleNext1D(sg) * triangleCount), triangleCount - 1);

        uint2 packed = emissiveSampler._emissivePower.triangleAliasTable[triangleIndex];
        float threshold = f16tof32(packed.x >> 16u);
        uint selectAbove = ((packed.x & 0xFFFFu) << 8u) | ((packed.y >> 24u) & 0xFFu);
        uint selectBelow = packed.y & 0xFFFFFFu;

        // Test the threshold in the current table entry; pick one of the two options
        triangleIndex = (sampleNext1D(sg) >= threshold) ? selectAbove : selectBelow;

        float triangleSelectionPdf = gScene.lightCollection.fluxData[triangleIndex].flux * emissiveSampler._emissivePower.invWeightsSum;

        if (triangleSelectionPdf <= 1e-6f) return false;

        // Sample emissive triangle
        const EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);
        const float3 barycentrics = sample_triangle(sampleNext2D(sg));
        pls.posW = tri.getPosition(barycentrics);

        // Get Flux
        float2 uv = tri.getTexCoord(barycentrics);
        pls.flux = gScene.materials.evalEmissive(tri.materialID, uv) * tri.area;

        pls.posW = computeRayOrigin(pls.posW, tri.normal); // Offset as this light is real geometry
        pls.flux /= triangleSelectionPdf; // FIXME This is zero or very small in the first frame

        float lightDirPDF = 1.0;
        float3 woLocal = sample_cosine_hemisphere_concentric(sampleNext2D(sg), lightDirPDF);

        if (lightDirPDF <= 0.f)
            return false;

        pls.dir = fromLocalToWorld(tri.normal, woLocal);
        pls.flux *= M_PI; // Lambert Emitter

        return true;
    }
//#endif

    /** Creates a photon light sample from analytic point or spot lights
        \param[in,out] Random Number Generator
        \param[out] Photon light sample
        \param[out] Direction the photon should be dispatched
        \return true if light sample is valid
    */
    bool generateAnalyticLightSample(inout SampleGenerator sg, out PhotonLightSample pls)
    {
        pls = {}; // Default initialization to avoid divergence at returns.
        const uint lightCount = gScene.getLightCount();
        if (lightCount == 0)
            return false;
        // Uniformly sample one of the analytic lights
        float rndLightIdx = float(lightCount) * sampleNext1D(sg);
        uint lightIdx = min(uint(floor(rndLightIdx)), lightCount - 1);
        LightData lightData = gScene.getLight(lightIdx);

        // Only Point(+ Spot) lights are supported (due to RTXDI)
        if (lightData.type != uint(LightType::Point))
            return false;
        float invLightPdf;
        float3 woLocal;
        float3 rnd = sampleNext3D(sg);
        rnd.z = 0;
        float spotAngle = lightData.cosOpeningAngle;
        if (lightData.openingAngle < M_2PI) // If the opening angle is smaller than 2PI it is a spotlight
        {
            spotAngle = cos(lightData.openingAngle - lightData.penumbraAngle * rnd.z);
            woLocal = sample_cone(rnd.xy, spotAngle);
        }
        else // Pointlight
        {
            woLocal = sample_sphere(rnd.xy);
        }
        invLightPdf = M_2PI * (1.f - spotAngle);

        if (invLightPdf <= 0.f)
            return false;

        invLightPdf *= lightCount; // Take the select pdf into account

        // Fill the photon light struct
        pls.posW = lightData.posW;
        pls.flux = lightData.intensity * invLightPdf;
        pls.dir = fromLocalToWorld(lightData.dirW, woLocal);
        return true;
    }

    /** Return the probabilities for selecting different light types.
        \param[out] p Probabilities.
    */
    void getLightTypeSelectionProbabilities(out float p[3])
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }

    float getEnvMapSelectionProbability()   { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyicSelectionProbability()  { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

    /** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
    bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
    {
        float p[3];
        getLightTypeSelectionProbabilities(p);

        float u = sampleNext1D(sg);

        [unroll]
        for (lightType = 0; lightType < 3; ++lightType)
        {
            if (u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }

        lightType = {};
        pdf = {};

        return false;
    }

    /** Samples a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample(inout SampleGenerator sg, out PhotonLightSample ls)
    {
        ls = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightSamplerType::EnvMap) valid = generateEnvMapSample(sg, ls);
        if (kUseEmissiveLights && lightType == (uint)LightSamplerType::Emissive)
        {
            valid = generateEmissiveSample(sg, ls);
        }
        if (kUseAnalyticLights && lightType == (uint)LightSamplerType::Analytic)
        {
            valid = generateAnalyticLightSample(sg, ls);
        }
        if (!valid) return false;

        // Account for light type selection.
        // ls.pdf *= selectionPdf;
        ls.flux /= selectionPdf;

        return true;
    }
};
