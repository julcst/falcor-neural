#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EmissiveLightSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"

using namespace Falcor;

class TracePhotons : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(TracePhotons, "TracePhotons", "Insert pass description here.");

    static ref<TracePhotons> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<TracePhotons>(pDevice, props);
    }

    TracePhotons(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void prepareVars();
    void prepareLightingStructure(RenderContext* pRenderContext);

    // Internal state
    ref<Scene>                      mpScene;                    ///< The current scene, or nullptr if no scene loaded.
    ref<SampleGenerator>            mpSampleGenerator;          ///< GPU pseudo-random sample generator.
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler;    ///< Emissive light sampler or nullptr if not used.

    // Configuration
    uint mPhotonCount = 1000000;            ///< Number of photons to trace per frame.
    uint mMaxBounces = 5;                   ///< Max number of bounces per photon.
    float mGlobalRejectionProb = 0.0f;      ///< Rejection probability

    // Runtime data
    /// Frame count since scene was loaded.
    uint mFrameCount = 0;

    // Ray tracing program.
    struct
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;
    } mTracer;
};
