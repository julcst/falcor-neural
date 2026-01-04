import RenderPasses.TraceQueries.Query;
import RenderPasses.NRC.NRC;

StructuredBuffer<Query> gQueries;
StructuredBuffer<NRCInput> gNRCInput;

// Output textures for Query fields
RWTexture2D<float4> gQueryPosition;
RWTexture2D<float4> gQueryViewDir;
RWTexture2D<float4> gQueryEmission;
RWTexture2D<float4> gQueryNormal;
RWTexture2D<float4> gQueryThroughput;
RWTexture2D<uint> gQueryGeomID;

// Output textures for NRCInput fields
RWTexture2D<float4> gNRCPosition;
RWTexture2D<float> gNRCRoughness;
RWTexture2D<float4> gNRCWo;
RWTexture2D<float4> gNRCWn;
RWTexture2D<float4> gNRCDiffuse;
RWTexture2D<float4> gNRCSpecular;

cbuffer CB
{
    uint2 gFrameDim;
};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gFrameDim)) return;
    
    uint idx = pixel.y * gFrameDim.x + pixel.x;
    
    // Output Query fields
    Query query = gQueries[idx];
    
#if is_valid_gQueryPosition
    gQueryPosition[pixel] = float4(query.position, 1.0);
#endif

#if is_valid_gQueryViewDir
    gQueryViewDir[pixel] = float4(query.getViewDir(), 1.0);
#endif

#if is_valid_gQueryEmission
    gQueryEmission[pixel] = float4(query.getEmission(), 1.0);
#endif

#if is_valid_gQueryNormal
    gQueryNormal[pixel] = float4(query.getNormal(), 1.0);
#endif

#if is_valid_gQueryThroughput
    gQueryThroughput[pixel] = float4(query.getThroughput(), 1.0);
#endif

#if is_valid_gQueryGeomID
    gQueryGeomID[pixel] = query.getHitInfo().getInstanceID().index;
#endif
    
    // Output NRCInput fields
#if is_valid_gNRCPosition || is_valid_gNRCRoughness || is_valid_gNRCWo || is_valid_gNRCWn || is_valid_gNRCDiffuse || is_valid_gNRCSpecular
    NRCInput nrc = gNRCInput[idx];
    
#if is_valid_gNRCPosition
    gNRCPosition[pixel] = float4(nrc.position, 1.0);
#endif

#if is_valid_gNRCRoughness
    gNRCRoughness[pixel] = nrc.roughness;
#endif

#if is_valid_gNRCWo
    gNRCWo[pixel] = float4(nrc.wo, 1.0);
#endif

#if is_valid_gNRCWn
    gNRCWn[pixel] = float4(nrc.wn, 1.0);
#endif

#if is_valid_gNRCDiffuse
    gNRCDiffuse[pixel] = float4(nrc.diffuse, 1.0);
#endif

#if is_valid_gNRCSpecular
    gNRCSpecular[pixel] = float4(nrc.specular, 1.0);
#endif
#endif
}
