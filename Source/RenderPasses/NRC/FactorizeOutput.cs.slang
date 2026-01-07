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
    
    gOutput[idx].radiance = clamp(out.radiance / in.weight(), 0.0f, 1e3f); // Clamp to safe range to avoid divergence
    // NOTE: fireflies > 1e3 lead to training loss evaluating to nan
}
