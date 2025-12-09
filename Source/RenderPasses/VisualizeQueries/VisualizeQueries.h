#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

using namespace Falcor;

class VisualizeQueries : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(VisualizeQueries, "VisualizeQueries", "Visualizes SPPM queries as disks.");

    static ref<VisualizeQueries> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<VisualizeQueries>(pDevice, props);
    }

    VisualizeQueries(ref<Device> pDevice, const Properties& props);

    virtual void setProperties(const Properties& props) override;
    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

private:
    ref<Scene> mpScene;
    ref<GraphicsState> mpGraphicsState;
    ref<Program> mpProgram;
    ref<ProgramVars> mpVars;
    ref<RasterizerState> mpRasterState;
    ref<DepthStencilState> mpDepthStencilState;
    ref<Vao> mpVao;

    // Config
    float mRadiusScale = 1.0f;
    bool mVisualizeCaustics = false;
    
    uint32_t mQueryCount = 0;
};
