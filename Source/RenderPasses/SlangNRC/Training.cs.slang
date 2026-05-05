#include "Utils/Math/MathConstants.slangh"

import RTXNS.MLP;
import RTXNS.LinearOps;
import RTXNS.Activation;
import RTXNS.PCG32;
import RTXNS.Utils;
import RTXNS.Loss;
import RTXNS.LinearOps;
import RTXNS.CooperativeVectorAutoDiff;
import RTXNS.CooperativeVectorFunctions;
import RTXNS.CooperativeVectorDerivatives;
import NRCCompat;
import Config;
import Encodings;
import Utils.Color.ColorHelpers;

using MLPConfig;

StructuredBuffer<NRCInput> gTrainInput;
StructuredBuffer<NRCOutput> gTrainTarget;
ByteAddressBuffer gParams;
RWByteAddressBuffer gParamGrads;
ByteAddressBuffer gEncodingParams;
RWByteAddressBuffer gEncodingParamGrads;

cbuffer CB
{
    uint gBatchOffset;
    uint gBatchSize;
    uint gCurrentStep;
}

float sampleNormalizedLossScale(uint sampleCount)
{
    return 1.0 / float(max(1u, sampleCount));
}

struct RelativeL2Luminance<T : __BuiltinFloatingPointType> : rtxns::mlp::ILoss<T, 3>
{
    static CoopVec<T, 3> value(CoopVec<T, 3> target, CoopVec<T, 3> predicted, CoopVec<T, 3> scale)
    {
        let lum = luminance(rtxns::VectorFromCoopVec(predicted));
        let denom = lum * lum + 1e-2;
        return scale * (predicted - target) * (predicted - target) * T(1 / denom);
    }

    static CoopVec<T, 3> deriv(CoopVec<T, 3> target, CoopVec<T, 3> predicted, CoopVec<T, 3> scale)
    {
        let lum = luminance(rtxns::VectorFromCoopVec(predicted));
        let denom = lum * lum + 1e-2;
        return scale * (predicted - target) * T(2.f / denom);
    }
};

[numthreads(64, 1, 1)]
void trainMain(uint3 tid : SV_DispatchThreadID)
{
    uint sampleIndex = tid.x;
    if (sampleIndex >= gBatchSize)
        return;

    uint idx = gBatchOffset + sampleIndex;

    let input = gTrainInput[idx];
    let rawTarget = gTrainTarget[idx].radiance;
    // Inline factorization: divide by sample weight when configured, clamp to safe range
#if USE_FACTORISATION
    let target = clamp(rawTarget / input.weight(), 0.0f, 1e3f);
#else
    let target = rawTarget;
#endif
    let targetVec = rtxns::CoopVecFromVector<half>(target);

    let pred = forward(gEncodingParams, gParams, input);

    let lossScale = CoopVec<half, kOutputSize>(half(sampleNormalizedLossScale(gBatchSize)));
    let outputGrad = RelativeL2Luminance<half>::deriv(targetVec, pred, lossScale);

    var features = FeatureBuffer(gEncodingParams);
    var featureGrads = FeatureBufferDifferential(gEncodingParamGrads);
    var weights = rtxns::mlp::MatrixBiasBuffer(gParams);
    var weightsGrads = rtxns::mlp::MatrixBiasBufferDifferential(gParamGrads);
    bwd_diff(forward)(DifferentialPtrPair<FeatureBuffer>(features, featureGrads), DifferentialPtrPair<rtxns::mlp::MatrixBiasBuffer>(weights, weightsGrads), input, outputGrad);
}