import RTXNS.Utils;
import Config;

using MLPConfig;

uint hashGridIndex(int2 cell, uint level)
{
	uint x = uint(cell.x);
	uint y = uint(cell.y);
	uint h = x * 0x8da6b343u;
	h ^= y * 0xd8163841u;
	h ^= level * 0xcb1ab31fu;
	return h & (kHashTableSize - 1u);
}

uint levelResolution(uint level)
{
	float scale = pow(kHashPerLevelScale, float(level));
	float resolution = float(kHashBaseResolution) * scale;
	return max(2u, uint(resolution + 0.5));
}

float loadEncodingParam(ByteAddressBuffer params, uint elementIndex)
{
	return float(params.Load<half>(elementIndex * sizeof(half)));
}

void addEncodingGrad(RWByteAddressBuffer grads, uint elementIndex, float value)
{
	grads.InterlockedAddF32(elementIndex * sizeof(float), value);
}

CoopVec<half, kInputSize> encodeHashGrid(ByteAddressBuffer params, float2 uv)
{
	CoopVec<half, kInputSize> encoded = CoopVec<half, kInputSize>(half(0.0));

	float2 uvClamped = saturate(uv);

	[ForceUnroll]
	for (uint level = 0u; level < kHashLevels; ++level)
	{
		uint resolution = levelResolution(level);
		float2 scaled = uvClamped * float(resolution - 1u);

		int2 p0 = int2(floor(scaled));
		int2 p1 = p0 + int2(1, 0);
		int2 p2 = p0 + int2(0, 1);
		int2 p3 = p0 + int2(1, 1);

		float2 fracCoord = frac(scaled);
		float w00 = (1.0 - fracCoord.x) * (1.0 - fracCoord.y);
		float w10 = fracCoord.x * (1.0 - fracCoord.y);
		float w01 = (1.0 - fracCoord.x) * fracCoord.y;
		float w11 = fracCoord.x * fracCoord.y;

		uint h00 = hashGridIndex(p0, level);
		uint h10 = hashGridIndex(p1, level);
		uint h01 = hashGridIndex(p2, level);
		uint h11 = hashGridIndex(p3, level);

		uint levelBase = level * kHashTableSize * kHashFeaturesPerLevel;
		uint outputBase = level * kHashFeaturesPerLevel;

		[ForceUnroll]
		for (uint f = 0u; f < kHashFeaturesPerLevel; ++f)
		{
			uint idx00 = levelBase + h00 * kHashFeaturesPerLevel + f;
			uint idx10 = levelBase + h10 * kHashFeaturesPerLevel + f;
			uint idx01 = levelBase + h01 * kHashFeaturesPerLevel + f;
			uint idx11 = levelBase + h11 * kHashFeaturesPerLevel + f;

			float v =
				w00 * loadEncodingParam(params, idx00) +
				w10 * loadEncodingParam(params, idx10) +
				w01 * loadEncodingParam(params, idx01) +
				w11 * loadEncodingParam(params, idx11);

			encoded[outputBase + f] = half(v);
		}
	}

	return encoded;
}

void accumulateHashGridGrads(RWByteAddressBuffer grads, float2 uv, CoopVec<half, kInputSize> dEncoded)
{
	float2 uvClamped = saturate(uv);

	[ForceUnroll]
	for (uint level = 0u; level < kHashLevels; ++level)
	{
		uint resolution = levelResolution(level);
		float2 scaled = uvClamped * float(resolution - 1u);

		int2 p0 = int2(floor(scaled));
		int2 p1 = p0 + int2(1, 0);
		int2 p2 = p0 + int2(0, 1);
		int2 p3 = p0 + int2(1, 1);

		float2 fracCoord = frac(scaled);
		float w00 = (1.0 - fracCoord.x) * (1.0 - fracCoord.y);
		float w10 = fracCoord.x * (1.0 - fracCoord.y);
		float w01 = (1.0 - fracCoord.x) * fracCoord.y;
		float w11 = fracCoord.x * fracCoord.y;

		uint h00 = hashGridIndex(p0, level);
		uint h10 = hashGridIndex(p1, level);
		uint h01 = hashGridIndex(p2, level);
		uint h11 = hashGridIndex(p3, level);

		uint levelBase = level * kHashTableSize * kHashFeaturesPerLevel;
		uint outputBase = level * kHashFeaturesPerLevel;

		[ForceUnroll]
		for (uint f = 0u; f < kHashFeaturesPerLevel; ++f)
		{
			float g = dEncoded[outputBase + f].toFloat();

			uint idx00 = levelBase + h00 * kHashFeaturesPerLevel + f;
			uint idx10 = levelBase + h10 * kHashFeaturesPerLevel + f;
			uint idx01 = levelBase + h01 * kHashFeaturesPerLevel + f;
			uint idx11 = levelBase + h11 * kHashFeaturesPerLevel + f;

			addEncodingGrad(grads, idx00, g * w00);
			addEncodingGrad(grads, idx10, g * w10);
			addEncodingGrad(grads, idx01, g * w01);
			addEncodingGrad(grads, idx11, g * w11);
		}
	}
}
