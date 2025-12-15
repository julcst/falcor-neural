#pragma once

#include "Utils/Math/MathConstants.slangh"
#include "Utils/HostDeviceShared.slangh"

#ifndef HOST_CODE
import Utils.Math.FormatConversion;
#endif

BEGIN_NAMESPACE_FALCOR

struct DebugCounters
{
    uint IntersectorCalls;
    uint Accumulations;
};

// Mutated multiple times per frame for each photon
// Reset every frame
struct Accumulator {
    float3 flux;
    uint collectedPhotons; // Optimal 16 byte alignment
#ifndef HOST_CODE
    __init() {
        flux = float3(0.0f);
        collectedPhotons = 0u;
    }

    __init(uint4 raw) {
        flux = asfloat(raw.xyz);
        collectedPhotons = raw.w;
    }

    __init(float3 flux, uint collectedPhotons) {
        this.flux = flux;
        this.collectedPhotons = collectedPhotons;
    }
#endif
};

struct DoubleAccumulator {
    Accumulator global;
    Accumulator caustic;
};

// Mutated once per frame after accumulation
// Reset only on camera/scene change
struct QueryState {
    float3 globalFlux;
    float globalCollectedPhotons;

    float3 causticFlux;
    float causticCollectedPhotons;

    uint totalPhotonCountAtBirth;
    float globalRadius; // TODO: Maintain separate radius for caustics
    float causticRadius;
    uint revision; // To identify changes

#ifndef HOST_CODE 
    __init(uint photonCountAtBirth, float globalRadius, float causticRadius, uint rev) {
        this.globalFlux = float3(0.0f);
        this.globalCollectedPhotons = 0.0f;
        this.causticFlux = float3(0.0f);
        this.causticCollectedPhotons = 0.0f;
        this.totalPhotonCountAtBirth = photonCountAtBirth;
        this.globalRadius = globalRadius;
        this.causticRadius = causticRadius;
        this.revision = rev;
    }

    [mutating]
    void applyRadiusReductionGlobal(Accumulator acc, float alpha) {
        if (acc.collectedPhotons == 0u) return; // No photons collected, no radius reduction needed
        const float newCollectedPhotons = globalCollectedPhotons + float(acc.collectedPhotons) * alpha;
        const float factor = newCollectedPhotons / (globalCollectedPhotons + float(acc.collectedPhotons));
        globalCollectedPhotons = newCollectedPhotons;
        globalRadius *= sqrt(factor);
        globalFlux += acc.flux;
        globalFlux *= factor;
    }

    [mutating]
    void applyRadiusReductionCaustic(Accumulator acc, float alpha) {
        if (acc.collectedPhotons == 0u) return; // No photons collected, no radius reduction needed
        const float newCollectedPhotons = causticCollectedPhotons + float(acc.collectedPhotons) * alpha;
        const float factor = newCollectedPhotons / (causticCollectedPhotons + float(acc.collectedPhotons));
        causticCollectedPhotons = newCollectedPhotons;
        causticRadius *= sqrt(factor);
        causticFlux += acc.flux;
        causticFlux *= factor;
    }

    [mutating]
    void applyRadiusReduction(DoubleAccumulator acc, float alphaGlobal, float alphaCaustic) {
        applyRadiusReductionGlobal(acc.global, alphaGlobal);
        applyRadiusReductionCaustic(acc.caustic, alphaCaustic);
    }

    float3 getRadiance(uint totalPhotonCount) {
        let photonCount = totalPhotonCount - totalPhotonCountAtBirth;
        let flux = globalFlux / (globalRadius * globalRadius) + causticFlux / (causticRadius * causticRadius);
        let weight = 1.0f / (photonCount * M_PI);
        return flux * weight;
    }

    float3 getCausticRadiance(uint totalPhotonCount) {
        let photonCount = totalPhotonCount - totalPhotonCountAtBirth;
        let flux = causticFlux / (causticRadius * causticRadius);
        let weight = 1.0f / (photonCount * M_PI);
        return flux * weight;
    }

    float3 getGlobalRadiance(uint totalPhotonCount) {
        let photonCount = totalPhotonCount - totalPhotonCountAtBirth;
        let flux = globalFlux / (globalRadius * globalRadius);
        let weight = 1.0f / (photonCount * M_PI);
        return flux * weight;
    }
#endif
};

#ifndef HOST_CODE
float sqr(half x) {
    let fx = float(x);
    return fx * fx;
}
#endif

// Holds only the geometric information about the query for
// high bandwith intersection tests
struct Sphere {
    float3 center;
    float16_t globalR; // Half for 16B alignment
    float16_t causticR;
    // TODO: Is this ok?

#ifndef HOST_CODE
    __init(float3 center, float globalR, float causticR) {
        this.center = center;
        this.globalR = half(globalR);
        this.causticR = half(causticR);
    }

    bool mayContain(float3 p) {
        let diff = p - center;
        let dist2 = dot(diff, diff);
        return dist2 < sqr(max(globalR, causticR));
    }

    bool contains(float3 p, bool isCaustic) {
        let diff = p - center;
        let dist2 = dot(diff, diff);
        return dist2 < (isCaustic ? sqr(causticR) : sqr(globalR));
    }
#endif
};

// TODO: What is worse high payload size or duplicate fetching of material data?
struct PhotonSearchPayload {
    uint4 hit;
    DoubleAccumulator accum;

#ifndef HOST_CODE
    __init(uint4 hit) {
        this.hit = hit;
        this.accum = {};
    }
#endif
};

struct PhotonIsectAttribs {
    float3 wi;
    bool isCaustic;
    float3 flux;
};

struct QuerySearchPayload {
    float3 flux;
    bool isCaustic;
    float3 normal;
};

struct QueryIsectAttribs {
    Sphere sphere;
};

END_NAMESPACE_FALCOR
