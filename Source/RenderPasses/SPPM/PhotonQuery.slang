module PhotonQuery;

public struct PhotonQuery
{
    public float3 position;
    public float radius;
    public float3 normal;
    float normalPad; // unused, keeps 16B alignment
    public float3 diffuseReflectance;
    float diffusePad; // unused, keeps 16B alignment
    public float3 viewDir;
    float viewPad; // unused, keeps 16B alignment
    public uint pixelIndex;
    public uint valid;
    uint2 _padding;
};
