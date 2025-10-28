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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EmissiveLightSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Core/API/RtAccelerationStructure.h"
#include "Core/Pass/ComputePass.h"
#include "Core/Pass/FullScreenPass.h"
#include "SPPMDebug.slangh"

using namespace Falcor;

class SPPM : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(SPPM, "SPPM", "Insert pass description here.");

    static ref<SPPM> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<SPPM>(pDevice, props);
    }

    SPPM(ref<Device> pDevice, const Properties& props);

    Properties getProperties() const override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void traceQueries(RenderContext* pRenderContext, const RenderData& renderData);
    void buildQueryAcceleration(RenderContext* pRenderContext);
    void resolveQueries(RenderContext* pRenderContext, const RenderData& renderData);
    void tracePhotonsPass(RenderContext* pRenderContext, const RenderData& renderData, bool analyticOnly, bool buildAS);
    void prepareLightingStructure(RenderContext* pRenderContext);
    void intersectPhotonsPass(RenderContext* pRenderContext, uint photonHitCount);
    // Uniform grid neighbor search
    void buildQueryGrid(RenderContext* pRenderContext);
    void accumulateByGrid(RenderContext* pRenderContext, uint photonHitCount);
    
    ///// Internal state /////

    /// Current scene.
    ref<Scene> mpScene;
    /// GPU sample generator.
    ref<SampleGenerator> mpSampleGenerator;

    std::unique_ptr<EmissiveLightSampler> mpEmissiveLightSampler; // Emissive triangle light sampler

    uint mFrameCount = 0;
    uint mDebugMode = 1; // 0: flux resolve, 1: query valid, 2: normals, 3: diffuse, 4: hit count
    uint mPrevDebugMode = mDebugMode;
    bool mUseRTXAccumulation = false; // Toggle: use RTX any-hit accumulation vs grid neighbor search

    // Light state
    bool mHasLights = false;           // True if the scene has any light sources
    bool mHasAnalyticLights = false;   // True if there are analytic lights
    bool mMixedLights = false;         // True if analytic and emissive lights are in the scene

    // Photon Distribution
    uint mPhotonMaxBounces = 10;                        // Number of photon bounces
    float mGlobalPhotonRejection = 0.3f;                // Probability that a global photon is stored
    uint mNumDispatchedPhotons = 2000000;               // Number of photons dispatched
    uint2 mNumMaxPhotons = uint2(400000, 300000);       // Size of the photon buffer
    uint2 mNumMaxPhotonsUI = mNumMaxPhotons;            // For UI, as changing happens with a button
    bool mChangePhotonLightBufferSize = true;           // True if buffer size has changed
    float mASBuildBufferPhotonOverestimate = 1.15f;     // Guard percentage for AS building
    uint2 mCurrentPhotonCount = mNumMaxPhotons;
    float2 mPhotonRadius = float2(0.020f, 0.005f);      // Global/Caustic Radius.
    float mPhotonAnalyticRatio = 0.5f;                  // Analytic photon distribution ratio in a mixed light case. E.g. 0.3 -> 30% analytic, 70% emissive

    //
    // Render Passes/Programs
    //
    struct RayTraceProgramHelper
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;

        static const RayTraceProgramHelper create()
        {
            RayTraceProgramHelper r;
            r.pProgram = nullptr;
            r.pBindingTable = nullptr;
            r.pVars = nullptr;
            return r;
        }

    void initProgramVars(const ref<Device>& pDevice, const ref<Scene>& pScene, const ref<SampleGenerator>& pSampleGenerator);
    };

    struct ComputeProgramHelper
    {
        ref<ComputePass> pPass;

        void dispatch(RenderContext* pRenderContext, uint3 dims)
        {
            FALCOR_ASSERT(pPass);
            pPass->execute(pRenderContext, dims);
        }
    };

    // Ray tracing program.
    RayTraceProgramHelper mTracePhotonPass;
    RayTraceProgramHelper mTraceQueryPass;
    RayTraceProgramHelper mIntersectPass;

    ComputeProgramHelper mBuildQueryBoundsPass;
    ComputeProgramHelper mResolveQueryPass; // deprecated path
    ref<FullScreenPass> mpResolveFullScreen; // graphics path for resolve
    ref<Buffer> mpQueryBuffer;
    ref<Buffer> mpQueryAABBBuffer;
    ref<Buffer> mpQueryAccumulator;
    // Photon hit buffers for separate accumulation pass
    ref<Buffer> mpPhotonHits;
    ref<Buffer> mpPhotonHitCounter;
    ref<Buffer> mpPhotonHitCounterReadback;

    // Uniform grid data for queries
    ref<Buffer> mpGridHeads;     // int per cell, head index of linked list (-1 if empty)
    ref<Buffer> mpGridNext;      // int per query, next pointer in linked list
    uint3       mGridDim = uint3(0);
    float3      mGridMin = float3(0.0f);
    float       mCellSize = 0.0f;

    // Compute passes
    ref<ComputePass> mpBuildGridPass;
    ref<ComputePass> mpGridAccumPass;
    ref<Buffer> mpDebugCounters;
    ref<Buffer> mpDebugCountersReadback;
    SPPMCounters mLastCounters = {}; // snapshot for UI (host-side mirror of struct)
    ref<Buffer> mpQueryInstanceBuffer;
    ref<Buffer> mpQueryBlasStorage;
    ref<Buffer> mpQueryBlasScratch;
    ref<Buffer> mpQueryTlasStorage;
    ref<Buffer> mpQueryTlasScratch;
    ref<RtAccelerationStructure> mpQueryBLAS;
    ref<RtAccelerationStructure> mpQueryTLAS;

    uint2 mFrameDim = uint2(0);
    uint mQueryCount = 0;
};
