// Structured counters interface using a typed struct, shared between host/device.
#pragma once

#include "Utils/HostDeviceShared.slangh"

#ifndef HOST_CODE
import Utils.Math.PackedFormats;
#endif

#define COMPRESS_PHOTON_HITS // Uncomment to enable compression of photon hits

BEGIN_NAMESPACE_FALCOR

struct PhotonCounters
{
    uint PhotonsEmitted;      // photons launched
    uint PhotonStores;        // photons stored as hits in TracePhotons
};

struct PhotonHitUncompressed
{
    float3 posW;
    bool isCaustic;
    float3 flux;
    uint pad0;
    float3 wi; // TODO: Pack
    uint pad1;
    float3 normal;
    uint pad2;

#ifndef HOST_CODE 
    __init(float3 posW, float3 wi, float3 normal, float3 flux, bool isCaustic)
    {
        this.posW = posW;
        this.wi = wi;
        this.normal = normal;
        this.flux = flux;
        this.isCaustic = isCaustic;
        this.pad0 = 0;
        this.pad1 = 0;
        this.pad2 = 0;
    }

    float3 getFlux() 
    {
        return this.flux;
    }

    float3 getWi() 
    {
        return this.wi;
    }

    float3 getNormal() 
    {
        return this.normal;
    }
#endif
};

struct PhotonHitCompressed
{
    float3 posW;
    bool isCaustic;

    uint flux;
    uint wi;
    uint normal;
    uint pad0;

#ifndef HOST_CODE 
    __init(float3 posW, float3 wi, float3 normal, float3 flux, bool isCaustic)
    {
        this.posW = posW;
        this.wi = encodeNormal2x16(wi);
        this.normal = encodeNormal2x16(normal);
        this.flux = encodeLogLuvHDR(flux);
        this.isCaustic = isCaustic;
        this.pad0 = 0;
    }

    float3 getFlux() 
    {
        return decodeLogLuvHDR(this.flux);
    }

    float3 getWi() 
    {
        return decodeNormal2x16(this.wi);
    }

    float3 getNormal() 
    {
        return decodeNormal2x16(this.normal);
    }
#endif
};

#ifdef COMPRESS_PHOTON_HITS
typedef PhotonHitCompressed PhotonHit;
#else
typedef PhotonHitUncompressed PhotonHit;
#endif

END_NAMESPACE_FALCOR
