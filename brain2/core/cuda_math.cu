#ifdef USE_CUDA
#include "cuda_math.cuh"
#include <cmath>

namespace brain2 {

__global__ void matvec_add_kernel(const float* W, const float* x, float* out, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        float s = 0.f;
        const float* row = W + i * n;
        for (int j = 0; j < n; j++) {
            s += row[j] * x[j];
        }
        out[i] += s;
    }
}

__global__ void matvec_sub_kernel(const float* W, const float* x, float* out, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) {
        float dp = out[i]; // This is lr * dpre[i]
        float* row = (float*)W + i * n; // Casting away const since we update W in place
        for (int j = 0; j < n; j++) {
            row[j] -= dp * x[j];
        }
    }
}

__global__ void som_update_kernel(float* weights, const float* input, int bmu, float eff_lr, float r2, int n_neurons, int n_dims, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_neurons) {
        float dr = (float)(i / cols) - (float)(bmu / cols);
        float dc = (float)(i % cols) - (float)(bmu % cols);
        float d2 = dr * dr + dc * dc;
        float h = expf(-d2 / r2);
        
        if (h >= 1e-4f) {
            float* w = weights + i * n_dims;
            float sc = eff_lr * h;
            for (int j = 0; j < n_dims; j++) {
                w[j] += sc * (input[j] - w[j]);
            }
        }
    }
}

__global__ void som_distances_kernel(const float* weights, const float* input, float* dists, int n_neurons, int n_dims) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_neurons) {
        float s = 0.f;
        const float* w = weights + i * n_dims;
        for (int j = 0; j < n_dims; j++) {
            float d = input[j] - w[j];
            s += d * d;
        }
        dists[i] = s;
    }
}

void cuda_matvec_add(const float* W, const float* x, float* out, int m, int n) {
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    matvec_add_kernel<<<blocks, threads>>>(W, x, out, m, n);
}

void cuda_matvec_sub(const float* W, const float* x, float* out, int m, int n) {
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    matvec_sub_kernel<<<blocks, threads>>>(W, x, out, m, n);
}

void cuda_som_update(float* weights, const float* input, int bmu, float eff_lr, float r2, int n_neurons, int n_dims, int cols) {
    int threads = 256;
    int blocks = (n_neurons + threads - 1) / threads;
    som_update_kernel<<<blocks, threads>>>(weights, input, bmu, eff_lr, r2, n_neurons, n_dims, cols);
}

void cuda_som_distances(const float* weights, const float* input, float* dists, int n_neurons, int n_dims) {
    int threads = 256;
    int blocks = (n_neurons + threads - 1) / threads;
    som_distances_kernel<<<blocks, threads>>>(weights, input, dists, n_neurons, n_dims);
}

void cuda_device_synchronize() {
    cudaDeviceSynchronize();
}

} // namespace brain2
#endif
