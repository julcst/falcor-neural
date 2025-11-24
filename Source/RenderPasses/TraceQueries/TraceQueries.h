#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
using namespace Falcor;

class TraceQueries : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(TraceQueries, "TraceQueries", "Perform simple GPU trace queries for testing.");

    static ref<TraceQueries> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<TraceQueries>(pDevice, props);
    }

    TraceQueries(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual void setProperties(const Properties& props) override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void prepareVars();

    // Internal state
    ref<Scene>                      mpScene;                    ///< The current scene, or nullptr if no scene loaded.
    ref<SampleGenerator>            mpSampleGenerator;          ///< GPU pseudo-random sample generator.

    // Ray tracing program state (program, SBT, vars)
    struct
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;
    } mTracer;

    // Resources written by the trace query shader. The render graph allocates these buffers and
    // they are accessed via `renderData.getResource()` at execute time.

    // Runtime state
    uint32_t mFrameCount = 0;                 ///< Frame counter since last scene change.
};
