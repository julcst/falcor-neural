#pragma once

#include "Utils/Math/MathConstants.slangh"
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

// Mutated multiple times per frame for each photon
// Reset every frame
struct Accumulator {
    float3 flux;
    uint collectedPhotons; // Optimal 16 byte alignment
#ifndef HOST_CODE
    __init(uint4 raw) {
        flux = asfloat(raw.xyz);
        collectedPhotons = raw.w;
    }
#endif
};

// Mutated once per frame after accumulation
// Reset only on camera/scene change
struct QueryState {
    float3 totalFlux;
    float totalCollectedPhotons;

    uint totalPhotonCountAtBirth;
    float radius; // TODO: Maintain separate radius for caustics

#ifndef HOST_CODE 
    [mutating]
    void applyRadiusReduction(Accumulator acc, float alpha) {
        if (acc.collectedPhotons == 0u) return; // No photons collected, no radius reduction needed
        const float newCollectedPhotons = totalCollectedPhotons + float(acc.collectedPhotons) * alpha;
        const float factor = newCollectedPhotons / (totalCollectedPhotons + float(acc.collectedPhotons));
        totalCollectedPhotons = newCollectedPhotons;
        radius *= sqrt(factor);
        totalFlux += acc.flux;
        totalFlux *= factor;
    }

    float3 getRadiance(uint totalPhotonCount) {
        let photonCount = totalPhotonCount - totalPhotonCountAtBirth;
        let r2 = radius * radius;
        let weight = 1.0f / (photonCount * r2 * M_PI);
        return totalFlux * weight;
    }
#endif
};

// Holds only the geometric information about the query for
// high bandwith intersection tests
struct Sphere {
    float3 center;
    float r2; // Optimal 16 byte alignment

#ifndef HOST_CODE
    bool contains(float3 p) {
        let diff = p - center;
        let dist2 = dot(diff, diff);
        return dist2 < r2;
    }
#endif
};

END_NAMESPACE_FALCOR
