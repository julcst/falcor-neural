__device__ __forceinline__ constexpr float3 operator*(const float3& a, const float3& b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ __forceinline__ constexpr float3 operator+(const float3& a, const float3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ constexpr float4 operator*(const float a, const float4& b) {
    return {a * b.x, a * b.y, a * b.z, a * b.w};
}

__device__ __forceinline__ constexpr float4 operator+=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__device__ __forceinline__ constexpr float4 make_float4(const float3& a, float w) {
    return {a.x, a.y, a.z, w};
}

__global__ void inference_kernel(
    int2 dim,
    float* inferenceInput, 
    float3* inferenceThroughput,
    bool raw,
    float4* image,
    const network_precision_t* __restrict__ params
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    bool valid = (x < dim.x) && (y < dim.y);

    const int i = y * dim.x + x;
    const int idxIn = i * NRC_INPUT_SIZE;

    // Pack input into hvec
    tcnn::vec<NRC_INPUT_SIZE> nerf_in;
    #pragma unroll
    for (int j = 0; j < NRC_INPUT_SIZE; j++)
        nerf_in[j] = valid ? inferenceInput[idxIn + j] : 0.0f;

    // Call tiny-cuda-nn model. All 32 threads of the warp must be active here.
    tcnn::vec<NRC_OUTPUT_SIZE> nerf_out = model_fun(nerf_in, params);

    if (!valid) return; // All threads must be active until now

    auto inference = make_float3(nerf_out[0], nerf_out[1], nerf_out[2]);

    const auto throughput = inferenceThroughput[i];

    if (throughput.x <= 0.0f && throughput.y <= 0.0f && throughput.z <= 0.0f) return;

    if (raw) {
        image[i] = make_float4(inference, 1.0f);
    } else {
        const auto diffuse = make_float3(nerf_in[8], nerf_in[9], nerf_in[10]);
        const auto specular = make_float3(nerf_in[11], nerf_in[12], nerf_in[13]);
        image[i] = make_float4(inference * (diffuse + specular) * throughput, 1.0f);
    }
}