#include "Utils/Math/MathConstants.slangh"

module PhotonQuery;

import Scene.HitInfo;

public struct PhotonQuery
{
    public PackedHitInfo hit;

    public float3 position;
    public float radius;

    public float3 viewDir;
    public uint pixelIndex;

    bool isValid() {
        return HitInfo(hit).isValid();
    }
};
