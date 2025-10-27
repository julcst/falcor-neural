// Minimal resolve to validate compute pipeline stability.
RWTexture2D<float4> gOutput;

cbuffer CB
{
    uint2 gFrameDim;
    uint gFrameIndex;
    uint _pad;
    float gAccumScale;
}

[numthreads(64,1,1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    uint idx = tid.x;
    uint total = gFrameDim.x * gFrameDim.y;
    if (idx >= total) return;

    uint2 px = uint2(idx % gFrameDim.x, idx / gFrameDim.x);
    gOutput[px] = float4(0,0,0,1);
}
