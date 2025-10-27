// Fullscreen resolve and debug visualization of queries/accumulation.

import PhotonQuery;

// Buffers
StructuredBuffer<uint4> gQueryAccumulationBuf;
StructuredBuffer<PhotonQuery> gPhotonQueries;
StructuredBuffer<uint> gDebugCounters; // [0]=considered, [1]=candidates, [2]=accumulations
RWTexture2D<float4> gOutput;

cbuffer CB
{
    uint2 gFrameDim;
    uint gFrameIndex;
    uint gDebugMode;    // 0: flux sum (default), 1: query valid mask, 2: query normals, 3: query diffuse, 4: hit count heatmap
    float gAccumScale;
}

struct VSOut
{
    float4 posH : SV_Position;
    float2 texC : TEXCOORD;
};

// Helpers
float3 heatmap(float v)
{
    // Simple blue->green->red heatmap for v in [0,1]
    v = saturate(v);
    float3 c = float3(0);
    c.r = smoothstep(0.5, 1.0, v);
    c.g = (v < 0.5) ? smoothstep(0.0, 0.5, v) : smoothstep(1.0, 0.5, v);
    c.b = 1.0 - smoothstep(0.0, 0.5, v);
    return c;
}

// FullScreenPass provides a default vertex shader; we only need psMain.

float4 psMain(VSOut pin) : SV_Target
{
    uint2 px = uint2(min((uint)(pin.texC.x * gFrameDim.x), gFrameDim.x - 1),
                     min((uint)(pin.texC.y * gFrameDim.y), gFrameDim.y - 1));
    uint idx = px.x + px.y * gFrameDim.x;

    PhotonQuery q = gPhotonQueries[idx];
    uint4 acc = gQueryAccumulationBuf[idx];

    if (gDebugMode == 1)
    {
        // Query valid mask
        return q.valid != 0 ? float4(1,1,1,1) : float4(0,0,0,1);
    }
    else if (gDebugMode == 2)
    {
        // World normal visualization
        float3 n = normalize(q.normal);
        return float4(0.5 * (n + 1.0), 1);
    }
    else if (gDebugMode == 3)
    {
        // Diffuse albedo
        return float4(q.diffuseReflectance, 1);
    }
    else if (gDebugMode == 4)
    {
        // Hit count heatmap (log scaled)
        float w = (float)acc.w;
        float v = (w <= 0.0) ? 0.0 : saturate(log2(1.0 + w) / 8.0);
        return float4(heatmap(v), 1);
    }
    else if (gDebugMode == 5)
    {
        // Visualize debug counters (log-scaled) in RGB.
        uint c0 = gDebugCounters[0];
        uint c1 = gDebugCounters[1];
        uint c2 = gDebugCounters[2];
        float3 v = float3(log2(1.0 + (float)c0), log2(1.0 + (float)c1), log2(1.0 + (float)c2)) / 16.0;
        return float4(saturate(v), 1);
    }
    else
    {
        // Default: flux sum resolve
        float3 sum = float3(acc.xyz) / gAccumScale;
        float w = (float)acc.w;
        return float4(sum, w);
    }
}
