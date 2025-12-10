#pragma once

#include "Utils/HostDeviceShared.slangh"
#include "Utils/Math/MathConstants.slangh"

#ifndef HOST_CODE
import Scene.HitInfo;
import Scene.ShadingData;
import Utils.Math.PackedFormats;
#endif

#define COMPRESS_QUERIES // Uncomment to enable compression of queries

BEGIN_NAMESPACE_FALCOR

struct QueryUncompressed
{
    uint4 hit;

    float3 position;
    uint revision; // To identify changes fast

    float3 viewDir;

    float3 normal;

    float3 emission; ///< Emission acumulated along the path from the camera to the query point, including at the query

    float3 throughput; ///< Throughput acumulated along the path from the camera to the query point

#ifndef HOST_CODE
    __init(uint4 hit, uint rev, ShadingData sd, float3 emission, float3 throughput)
    {
        this.hit = hit;
        this.position = sd.posW;
        this.revision = rev;
        this.viewDir = sd.V;
        this.normal = sd.faceN; // TODO: Use guide normal?
        this.emission = emission;
        this.throughput = throughput;
    }

    float3 transportRadianceToCamera(float3 radiance)
    {
        return radiance * throughput + emission;
    }

    float3 getViewDir()
    {
        return viewDir;
    }

    float3 getNormal()
    {
        return normal;
    }

    float3 getEmission()
    {
        return emission;
    }

    float3 getThroughput()
    {
        return throughput;
    }

    HitInfo getHitInfo() {
        return unpackHitInfo(hit);
    }

    bool isValid() {
        // NOTE: The isValid() function needs hit defines to be set
        // return getHitInfo().isValid();
        // Thus we use this simpler direct check, which works because type is encoded in the first component
        // hit.x = 0 => Type = None but not the other way around
        return hit.x != 0u;
    }
#endif
};

// TODO: Use
struct QueryCompressed
{
    uint4 hit;

    float3 position;
    uint revision; // To identify changes fast?

    uint viewDir;
    uint normal;
    uint emission;
    uint throughput;

#ifndef HOST_CODE
    __init(uint4 hit, uint rev, ShadingData sd, float3 emission, float3 throughput)
    {
        this.hit = hit;
        this.position = sd.posW;
        this.revision = rev;
        this.viewDir = encodeNormal2x16(sd.V);
        this.normal = encodeNormal2x16(sd.faceN); // TODO: Use guide normal?
        this.emission = encodeLogLuvHDR(emission);
        this.throughput = encodeLogLuvHDR(throughput);
    }

    float3 transportRadianceToCamera(float3 radiance)
    {
        return radiance * getThroughput() + getEmission();
    }

    float3 getViewDir()
    {
        return decodeNormal2x16(viewDir);
    }

    float3 getNormal()
    {
        return decodeNormal2x16(normal);
    }

    float3 getEmission()
    {
        return decodeLogLuvHDR(emission);
    }

    float3 getThroughput()
    {
        return decodeLogLuvHDR(throughput);
    }

    HitInfo getHitInfo() {
        return unpackHitInfo(hit);
    }

    bool isValid() {
        // NOTE: The isValid() function needs hit defines to be set
        // return getHitInfo().isValid();
        // Thus we use this simpler direct check, which works because type is encoded in the first component
        // hit.x = 0 => Type = None but not the other way around
        return hit.x != 0u;
    }
#endif
};

#ifdef COMPRESS_QUERIES
typedef QueryCompressed Query;
#else
typedef QueryUncompressed Query;
#endif

END_NAMESPACE_FALCOR
