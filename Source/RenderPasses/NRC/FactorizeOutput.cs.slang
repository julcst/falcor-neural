import NRC;

RWStructuredBuffer<NRCOutput> gOutput;
StructuredBuffer<NRCInput> gInput;

cbuffer CB
{
    uint gCount;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    let idx = dispatchThreadId.x;
    if (idx >= gCount) return;

    let out = gOutput[idx];
    let in = gInput[idx];
    
    gOutput[idx].radiance = out.radiance / in.weight();
}
