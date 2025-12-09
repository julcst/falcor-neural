#include "VisualizeQueries.h"
#include "Scene/TriangleMesh.h"
#include "../TraceQueries/Query.slang"
#include "../AccumulatePhotonsRTX/Structs.slang"

namespace
{
    const char kShaderFile[] = "RenderPasses/VisualizeQueries/VisualizeQueries.3d.slang";
    const char kQueryBuffer[] = "queries";
    const char kQueryStateBuffer[] = "queryStates";
    const char kOutputTexture[] = "output";

    const char kRadiusScale[] = "radiusScale";
    const char kVisualizeCaustics[] = "visualizeCaustics";
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VisualizeQueries>();
}

VisualizeQueries::VisualizeQueries(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    setProperties(props);
    mpGraphicsState = GraphicsState::create(pDevice);
    
    // Create rasterizer state
    RasterizerState::Desc rasterDesc;
    rasterDesc.setCullMode(RasterizerState::CullMode::None); // Draw both sides of disk
    mpRasterState = RasterizerState::create(rasterDesc);
    mpGraphicsState->setRasterizerState(mpRasterState);

    // Create depth stencil state
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthEnabled(true);
    dsDesc.setDepthWriteMask(false); // Read-only depth
    dsDesc.setDepthFunc(ComparisonFunc::LessEqual);
    mpDepthStencilState = DepthStencilState::create(dsDesc);
    mpGraphicsState->setDepthStencilState(mpDepthStencilState);

    // Create program
    ProgramDesc desc;
    desc.addShaderLibrary(kShaderFile).vsEntry("vsMain").psEntry("psMain");
    //desc.setShaderModel(ShaderModel::SM6_5);
    mpProgram = Program::create(pDevice, desc);
    mpGraphicsState->setProgram(mpProgram);

    // Create disk mesh
    auto pDisk = TriangleMesh::createDisk(1.0f, 16);
    auto pVb = pDevice->createStructuredBuffer(sizeof(TriangleMesh::Vertex), pDisk->getVertices().size(), ResourceBindFlags::Vertex, MemoryType::DeviceLocal, pDisk->getVertices().data());
    auto pIb = pDevice->createStructuredBuffer(sizeof(uint32_t), pDisk->getIndices().size(), ResourceBindFlags::Index, MemoryType::DeviceLocal, pDisk->getIndices().data());

    // Create VAO
    ref<VertexLayout> pLayout = VertexLayout::create();
    ref<VertexBufferLayout> pBufferLayout = VertexBufferLayout::create();
    pBufferLayout->addElement("POSITION", offsetof(TriangleMesh::Vertex, position), ResourceFormat::RGB32Float, 1, 0);
    pBufferLayout->addElement("NORMAL", offsetof(TriangleMesh::Vertex, normal), ResourceFormat::RGB32Float, 1, 1);
    pBufferLayout->addElement("TEXCOORD", offsetof(TriangleMesh::Vertex, texCoord), ResourceFormat::RG32Float, 1, 2);
    pLayout->addBufferLayout(0, pBufferLayout);

    Vao::BufferVec buffers{ pVb };
    mpVao = Vao::create(Vao::Topology::TriangleList, pLayout, buffers, pIb, ResourceFormat::R32Uint);
    mpGraphicsState->setVao(mpVao);
}

void VisualizeQueries::setProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kRadiusScale) mRadiusScale = value;
        else if (key == kVisualizeCaustics) mVisualizeCaustics = value;
        else logWarning("{} - Unknown property '{}'.", getClassName(), key);
    }
}

Properties VisualizeQueries::getProperties() const
{
    Properties props;
    props[kRadiusScale] = mRadiusScale;
    props[kVisualizeCaustics] = mVisualizeCaustics;
    return props;
}

RenderPassReflection VisualizeQueries::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    
    reflector.addInput(kQueryBuffer, "Buffer containing ray query results.")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::ShaderResource);
        
    reflector.addInput(kQueryStateBuffer, "Buffer containing query states (radius, flux).")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::ShaderResource);

    reflector.addOutput(kOutputTexture, "Output texture.")
        .texture2D(0, 0)
        .bindFlags(ResourceBindFlags::RenderTarget);
        
    return reflector;
}

void VisualizeQueries::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
}

void VisualizeQueries::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    const auto pQueryBuffer = renderData[kQueryBuffer]->asBuffer();
    const auto pQueryStateBuffer = renderData[kQueryStateBuffer]->asBuffer();
    const auto pOutputTexture = renderData.getTexture(kOutputTexture);
    
    if (!pQueryBuffer || !pQueryStateBuffer || !pOutputTexture) return;

    // Update vars
    if (!mpVars) mpVars = ProgramVars::create(mpDevice, mpProgram.get());

    auto var = mpVars->getRootVar();
    var["gQueries"] = pQueryBuffer;
    var["gQueryStates"] = pQueryStateBuffer;
    
    var["CB"]["gRadiusScale"] = mRadiusScale;
    var["CB"]["gVisualizeCaustics"] = mVisualizeCaustics;
    var["CB"]["gGlobalPhotonCount"] = renderData.getDictionary().getValue<uint32_t>("GlobalPhotonCount", 0);
    
    // Bind camera
    mpScene->getCamera()->bindShaderData(var["CB"]["gCamera"]);

    // Setup FBO
    auto pFbo = Fbo::create(mpDevice);
    pFbo->attachColorTarget(pOutputTexture, 0);
    
    mpGraphicsState->setFbo(pFbo);
    
    // Calculate element count
    uint32_t instanceCount = pQueryBuffer->getSize() / sizeof(Query);
    uint32_t indexCount = mpVao->getIndexBuffer()->getElementCount();
    
    // Draw
    pRenderContext->clearFbo(pFbo.get(), float4(0), 1.0f, 0);
    pRenderContext->drawIndexedInstanced(mpGraphicsState.get(), mpVars.get(), indexCount, instanceCount, 0, 0, 0);
}

void VisualizeQueries::renderUI(Gui::Widgets& widget)
{
    widget.var("Radius Scale", mRadiusScale, 0.1f, 10.0f);
    widget.checkbox("Visualize Caustics", mVisualizeCaustics);
}

void VisualizeQueries::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    if (mpScene)
    {
        mpProgram->addDefines(mpScene->getSceneDefines());
    }
}
