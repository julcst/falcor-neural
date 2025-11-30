import RenderPasses.TraceQueries.Query;
import RenderPasses.NRC.NRC;
import Utils.Math.HashUtils;

StructuredBuffer<Query> gInputQueries;
RWStructuredBuffer<Query> gOutputQueries;

StructuredBuffer<NRCInput> gInputNRC;
RWStructuredBuffer<NRCInput> gOutputNRC;

cbuffer CB
{
    uint gInputCount;
    uint gOutputCount;
    uint gFrameCount;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (gInputCount == 0) return;
    let idx = dispatchThreadId.x;
    if (idx >= gOutputCount) return;

    // Random index
    let inputIdx = pcg2d(uint2(idx, gFrameCount)).x % gInputCount;

    gOutputQueries[idx] = gInputQueries[inputIdx];

#if SUBSAMPLE_NRC
    gOutputNRC[idx] = gInputNRC[inputIdx];
#endif
}
