ByteAddressBuffer gAccumulator;
RWTexture2D<float4> gOutput;

cbuffer CB
{
    uint2 gFrameDim;
    bool gVisualizeHeatmap;
}

float3 heatmap(float v)
{
    // Simple blue->green->red heatmap for v in [0,1]
    v = saturate(v);
    var c = float3(0);
    c.r = smoothstep(0.5, 1.0, v);
    c.g = (v < 0.5) ? smoothstep(0.0, 0.5, v) : smoothstep(1.0, 0.5, v);
    c.b = 1.0 - smoothstep(0.0, 0.5, v);
    return c;
}

float mapTo01(uint x)
{
    // Map integer to [0,1] for visualization
    return log(float(x) + 1.0) / log(1000.0 + 1.0); // Assuming max ~1000 for better contrast
}

[numthreads(16,16,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    let px = tid.xy;
    if (any(px >= gFrameDim)) return;
    let idx = px.x + px.y * gFrameDim.x;

    let acc = gAccumulator.Load4Aligned(idx * sizeof(float4));
    let flux = asfloat(acc.xyz);
    let photons = acc.w;

    let radiance = flux;

    gOutput[px] = gVisualizeHeatmap ? float4(heatmap(mapTo01(photons)), 1.0) : float4(radiance, 1.0);
}

