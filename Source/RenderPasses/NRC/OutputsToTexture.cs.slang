import NRC;
import RenderPasses.TraceQueries.Query;

StructuredBuffer<NRCOutput> gInferenceOutput;
StructuredBuffer<Query> gInferenceQueries;
RWTexture2D<float4> gOutput;

cbuffer CB
{
    uint2 gResolution;
};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    let pixel = dispatchThreadId.xy;
    if (any(pixel >= gResolution)) return;

    let idx = pixel.y * gResolution.x + pixel.x;
    let out = gInferenceOutput[idx];
    let query = gInferenceQueries[idx];

    gOutput[pixel] = float4(out.radiance * query.throughput, 1.0);
}
