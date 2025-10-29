#include "Utils/Math/MathConstants.slangh"
#include "RenderPasses/SPPM/SPPMDebug.slangh"

import PhotonQuery;
import PhotonHit;

StructuredBuffer<PhotonHit>    gPhotonHits;      // [photonCount]
StructuredBuffer<PhotonQuery>  gPhotonQueries;   // [queryCount]
StructuredBuffer<int>          gGridCellQuery;   // [gridCells], single query index per cell (-1 if empty)
RWStructuredBuffer<uint4>      gQueryAccumulation; // fixed-point accumulation
RWStructuredBuffer<SPPMCounters> gCounters;     // single element counters struct

cbuffer CB
{
    uint   gPhotonCount;
    float  gAccumScale;
    uint3  gGridDim;
    float3 gGridMin;
    float  gCellSize;
    float  gMaxQueryRadius;
}

static uint flatten3D(uint3 c, uint3 dim)
{
    return c.x + dim.x * (c.y + dim.y * c.z);
}

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    if (idx >= gPhotonCount) return;

    InterlockedAdd(gCounters[0].PhotonsConsidered, 1);

    PhotonHit ph = gPhotonHits[idx];

    // Determine the integer cell range overlapped by the maximum query radius around the photon.
    float3 minP = ph.posW - gMaxQueryRadius;
    float3 maxP = ph.posW + gMaxQueryRadius;
    int3 minCell = int3(floor((minP - gGridMin) / gCellSize));
    int3 maxCell = int3(floor((maxP - gGridMin) / gCellSize));
    minCell = clamp(minCell, int3(0,0,0), int3(gGridDim) - 1);
    maxCell = clamp(maxCell, int3(0,0,0), int3(gGridDim) - 1);

    // Iterate all potentially overlapping cells, culling by distance to cell AABB.
    for (int z = minCell.z; z <= maxCell.z; ++z)
    {
        for (int y = minCell.y; y <= maxCell.y; ++y)
        {
            for (int x = minCell.x; x <= maxCell.x; ++x)
            {
                float3 cellMin = gGridMin + float3(x, y, z) * gCellSize;
                float3 cellMax = cellMin + gCellSize;
                float3 closest = clamp(ph.posW, cellMin, cellMax);
                float3 cd = closest - ph.posW;
                if (dot(cd, cd) > gMaxQueryRadius * gMaxQueryRadius)
                    continue; // Cell completely outside of the maximum query radius.

                uint cell = flatten3D(uint3(x,y,z), gGridDim);
                int qHead = gGridCellQuery[cell];
                if (qHead == -1) continue;

                uint qIdx = (uint)qHead;
                PhotonQuery q = gPhotonQueries[qIdx];
                if (q.valid == 0) continue;

                float3 d = ph.posW - q.position;
                float dist2 = dot(d, d);
                float r2 = q.radius * q.radius;
                InterlockedAdd(gCounters[0].Candidates, 1); // candidates
                if (dist2 <= r2)
                {
                    float3 wi = normalize(-d);
                    float NoL = max(0.0, dot(q.normal, wi));
                    float3 f = q.diffuseReflectance * (M_1_PI) * NoL;
                    float3 c = ph.flux * f;

                    uint pix = q.pixelIndex;
                    uint4 inc = uint4(
                        (uint)max(0.0, c.x * gAccumScale + 0.5),
                        (uint)max(0.0, c.y * gAccumScale + 0.5),
                        (uint)max(0.0, c.z * gAccumScale + 0.5),
                        1u);

                    InterlockedAdd(gQueryAccumulation[pix].x, inc.x);
                    InterlockedAdd(gQueryAccumulation[pix].y, inc.y);
                    InterlockedAdd(gQueryAccumulation[pix].z, inc.z);
                    InterlockedAdd(gQueryAccumulation[pix].w, inc.w);

                    InterlockedAdd(gCounters[0].Accumulations, 1); // accumulations
                }
            }
        }
    }
}
