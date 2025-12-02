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

static const uint kMaxTrials = 3;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (gInputCount == 0) return;
    let idx = dispatchThreadId.x;
    if (idx >= gOutputCount) return;

    // Random index
    var rand = pcg2d(uint2(idx, gFrameCount));
    var inputIdx = rand.x % gInputCount;

    for (uint trial = 0; trial < kMaxTrials && !gInputQueries[inputIdx].isValid(); ++trial)
    {
        // Draw new random index
        rand = pcg2d(rand);
        inputIdx = rand.x % gInputCount;
    }

    let q = gInputQueries[inputIdx];
    //if (!q.isValid()) return; // Reuse old query sample if we failed to find a valid one

    gOutputQueries[idx] = q;
#if SUBSAMPLE_NRC
    gOutputNRC[idx] = gInputNRC[inputIdx];
#endif
}
