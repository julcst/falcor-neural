#include "VisualizeQueries.h"
#include "Scene/TriangleMesh.h"
#include "../TraceQueries/Query.slang"
#include "../AccumulatePhotonsRTX/Structs.slang"

namespace
{
    const char kShaderFile[] = "RenderPasses/VisualizeQueries/VisualizeQueries.3d.slang";
    const char kQueryBuffer[] = "queries";
    const char kQueryStateBuffer[] = "queryStates";
    const char kColorBuffer[] = "colors";
    const char kOutputTexture[] = "output";

    const char kConstantRadius[] = "constantRadius";
    const char kMode[] = "mode";
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

    // Blending
    BlendState::Desc blendDesc;
    blendDesc.setRtBlend(0, true);
    blendDesc.setRtParams(0, BlendState::BlendOp::Add, BlendState::BlendOp::Add, BlendState::BlendFunc::SrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha, BlendState::BlendFunc::One, BlendState::BlendFunc::One);
    mpBlendState = BlendState::create(blendDesc);
    mpGraphicsState->setBlendState(mpBlendState);

    // Create program
    ProgramDesc desc;
    desc.addShaderLibrary(kShaderFile).vsEntry("vsMain").psEntry("psMain");
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
        if (key == kConstantRadius) mConstantRadius = value;
        else if (key == kMode) mMode = static_cast<Mode>(value.operator uint32_t());
        else if (key == "visualizeCaustics") mMode = value ? Mode::Caustic : Mode::Global; // backwards compatibility
        else logWarning("{} - Unknown property '{}'.", getClassName(), key);
    }
}

Properties VisualizeQueries::getProperties() const
{
    Properties props;
    props[kConstantRadius] = mConstantRadius;
    props[kMode] = static_cast<uint32_t>(mMode);
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
        .bindFlags(ResourceBindFlags::ShaderResource)
        .flags(RenderPassReflection::Field::Flags::Optional);

    reflector.addInput(kColorBuffer, "Optional float3 buffer for colors.")
        .rawBuffer(0)
        .bindFlags(ResourceBindFlags::ShaderResource)
        .flags(RenderPassReflection::Field::Flags::Optional);

    reflector.addOutput(kOutputTexture, "Output texture.")
        .texture2D(0, 0)
        .bindFlags(ResourceBindFlags::RenderTarget)
        .format(ResourceFormat::RGBA32Float);
        
    return reflector;
}

void VisualizeQueries::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
}

void VisualizeQueries::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    const auto pQueryBuffer = renderData.getBuffer(kQueryBuffer);
    const auto pQueryStateBuffer = renderData.getBuffer(kQueryStateBuffer);
    const auto pColorBuffer = renderData.getBuffer(kColorBuffer);
    const auto pOutputTexture = renderData.getTexture(kOutputTexture);
    
    if (!pQueryBuffer || !pOutputTexture) return;

    // Validate buffer sizes match
    uint32_t queryCount = pQueryBuffer->getSize() / sizeof(Query);
    if (pQueryStateBuffer)
    {
        uint32_t queryStateCount = pQueryStateBuffer->getSize() / sizeof(QueryState);
        FALCOR_ASSERT_EQ(queryCount, queryStateCount);
    }
    if (pColorBuffer)
    {
        uint32_t colorCount = pColorBuffer->getSize() / sizeof(float3);
        FALCOR_ASSERT_EQ(queryCount, colorCount);
    }

    // Resolve mode depending on available data
    Mode mode = mMode;
    if (!pQueryStateBuffer && mode != Mode::Constant)
    {
        logWarningOnce("VisualizeQueries: Query states missing, falling back to constant mode.");
        mode = Mode::Constant;
    }

    mpProgram->addDefine("USE_QUERY_STATE", pQueryStateBuffer ? "1" : "0");
    mpProgram->addDefine("USE_COLOR_BUFFER", pColorBuffer ? "1" : "0");

    // Update vars
    if (!mpVars) mpVars = ProgramVars::create(mpDevice, mpProgram.get());

    auto var = mpVars->getRootVar();
    var["gQueries"] = pQueryBuffer;
    if (pQueryStateBuffer) var["gQueryStates"] = pQueryStateBuffer;
    if (pColorBuffer) var["gQueryColors"] = pColorBuffer;
    
    var["CB"]["gConstantRadius"] = mConstantRadius;
    var["CB"]["gMode"] = static_cast<uint32_t>(mode);
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
    widget.var("Constant Radius", mConstantRadius, 0.001f, 1.0f);
    uint32_t mode = static_cast<uint32_t>(mMode);
    if (widget.dropdown("Mode", {
        {(uint32_t)Mode::Caustic, "Caustic Radiance/Radius"},
        {(uint32_t)Mode::Global, "Global Radiance/Radius"},
        {(uint32_t)Mode::MaxRadius, "Max Radius + Combined"},
        {(uint32_t)Mode::Constant, "Constant Radius + Color"},
    }, mode))
    {
        mMode = static_cast<Mode>(mode);
    }
}

void VisualizeQueries::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    if (mpScene)
    {
        mpProgram->addDefines(mpScene->getSceneDefines());
    }
}
