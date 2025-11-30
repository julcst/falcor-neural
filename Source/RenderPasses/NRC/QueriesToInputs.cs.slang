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

    let q = gQueries[idx];
    let in = NRCInput(q);
    gInputFloat[idx] = in;
}
