// Structured counters interface using a typed struct, shared between host/device.
#pragma once

#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

struct PhotonCounters
{
    uint PhotonsEmitted;      // photons launched
    uint PhotonStores;        // photons stored as hits in TracePhotons
};

struct PhotonHit
{
    float3 posW;
    bool isCaustic;
    float3 flux;
    uint pad0;
    float3 wi; // TODO: Pack
    uint pad1;

#ifndef HOST_CODE 
    __init(float3 posW, float3 wi, float3 flux, bool isCaustic)
    {
        this.posW = posW;
        this.wi = wi;
        this.flux = flux;
        this.isCaustic = isCaustic;
        this.pad0 = 0;
        this.pad1 = 0;
    }
#endif
};

END_NAMESPACE_FALCOR
