import PhotonQuery;
#include "RenderPasses/SPPM/SPPMDebug.slangh"

StructuredBuffer<PhotonQuery> gPhotonQueries; // [queryCount]
RWStructuredBuffer<int>      gGridHeads;     // [gridDim.x*gridDim.y*gridDim.z], initialized to -1
RWStructuredBuffer<int>      gGridNext;      // [queryCount]
RWStructuredBuffer<SPPMCounters> gCounters; // debug stats (single element)

cbuffer CB
{
    uint3 gGridDim;          // grid resolution
    float3 gGridMin;         // world-space min
    float  gCellSize;        // uniform cell size
    uint   gQueryCount;      // total number of queries
}

static uint flatten3D(uint3 c, uint3 dim)
{
    return c.x + dim.x * (c.y + dim.y * c.z);
}

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint qIdx = tid.x;
    if (qIdx >= gQueryCount) return;

    PhotonQuery q = gPhotonQueries[qIdx];
    if (q.valid == 0) { gGridNext[qIdx] = -1; return; }

    float3 rel = (q.position - gGridMin) / gCellSize;
    int3 ic = int3(floor(rel + 1e-6));
    ic = clamp(ic, int3(0,0,0), int3(gGridDim) - 1);
    uint cell = flatten3D(uint3(ic), gGridDim);

    int prev;
    InterlockedExchange(gGridHeads[cell], int(qIdx), prev);
    gGridNext[qIdx] = prev; // -1 if empty, otherwise previous head
    // Stats
    InterlockedAdd(gCounters[0].GridBinnedQueries, 1);
    if (prev == -1) InterlockedAdd(gCounters[0].GridNonEmptyCells, 1);
}
