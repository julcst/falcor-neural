// Structured counters interface using a typed struct, shared between host/device.
#pragma once

#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct PhotonCounters
{
    uint PhotonsDispatched;   // photons processed in grid/rt accumulation
    uint PhotonsEmitted;      // photons launched
    uint TotalBounces;       // total photon bounces
    uint PhotonStores;        // photons stored as hits in TracePhotons
};

struct PhotonHit
{
    float3 posW;
    float pad0;
    float3 flux;
    float pad1;
    float3 wi;
    float pad2;

#ifndef HOST_CODE 
    __init(float3 posW, float3 wi, float3 flux)
    {
        this.posW = posW;
        this.wi = wi;
        this.flux = flux;
    }
#endif
};

END_NAMESPACE_FALCOR
