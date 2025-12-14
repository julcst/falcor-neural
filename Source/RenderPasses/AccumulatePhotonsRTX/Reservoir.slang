import Utils.Sampling.SampleGeneratorInterface;

struct Reservoir<T: IArithmetic, let N: uint> {
    uint n; // number of samples stored in the reservoir
    T samples[N];

    __init() {
        n = 0;
    }

    [mutating]
    void addSample<S: ISampleGenerator>(T sample, inout S sg) {
        if (n < N) {
            samples[n] = sample;
        } else if (sampleNext1D(sg) < float(N) / float(n)) {
            let replaceIndex = sg.next() % N;
            samples[replaceIndex] = sample;
        }
        n++;
    }

    uint begin() {
        return 0;
    }

    uint end() {
        return min(n, N);
    }

    uint getTotalCount() {
        return n;
    }

    T get(uint index) {
        return samples[index];
    }

    float getWeight() {
        if (n == 0) return 0.0f;
        return float(n) / float(end());
    }
}