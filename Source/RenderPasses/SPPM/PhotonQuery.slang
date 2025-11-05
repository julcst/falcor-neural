module PhotonQuery;

import Scene.HitInfo;

public struct PhotonQuery
{
    public PackedHitInfo hit;
    public float3 position;
    public float radius;
    public float3 viewDir;
    float viewPad; // unused, keeps 16B alignment
    public uint pixelIndex;
    public uint valid;
    uint2 _padding;
};
