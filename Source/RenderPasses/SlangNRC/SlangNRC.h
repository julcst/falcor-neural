#pragma once

#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Core/Pass/ComputePass.h"

#include <memory>

using namespace Falcor;

class SlangNRC : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(SlangNRC, "SlangNRC", "Insert pass description here.");

    static constexpr const uint32_t BATCH_SIZE_GRANULARITY = 256;

    static ref<SlangNRC> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<SlangNRC>(pDevice, props);
    }

    SlangNRC(ref<Device> pDevice, const Properties& props);

    virtual void setProperties(const Properties& props) override;
    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override {}
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void createPasses();

    uint32_t mInferenceQueryCount = 0;
    uint32_t mInferenceSize = 0;
    uint32_t mTrainSize = 0;

    ref<ComputePass> mpResetPass;
    ref<ComputePass> mpTrainPass;
    ref<ComputePass> mpOptimizePass;
    ref<ComputePass> mpInferPass;

    uint32_t mOptimizeStep = 1;
    uint32_t mTrainSteps = 1;
    float mLearningRate = 1e-3f;
    bool mUseFactorization = true;
    bool mOutputRaw = false;
    bool mReset = true;
};
