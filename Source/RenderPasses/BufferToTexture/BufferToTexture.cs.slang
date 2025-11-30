StructuredBuffer<float3> gInput;
RWTexture2D<float4> gOutput;

cbuffer CB
{
    uint2 gResolution;
};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gResolution)) return;

    uint idx = pixel.y * gResolution.x + pixel.x;
    float3 val = gInput[idx];

    gOutput[pixel] = float4(val, 1.0);
}
