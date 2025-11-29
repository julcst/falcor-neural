import NRC;
import RenderPasses.TraceQueries.Query;

StructuredBuffer<Query> gQueries;
RWStructuredBuffer<NRCInput> gInputFloat;

cbuffer CB
{
    uint gCount;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    let idx = dispatchThreadId.x;
    if (idx >= gCount) return;

    let in = NRCInput(gQueries[idx]);
    gInputFloat[idx] = in;
}
