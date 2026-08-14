#pragma once

#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

namespace brain2 {

// C++ Allocator using CUDA Managed Memory
template<typename T>
struct ManagedAllocator {
    typedef T value_type;
    ManagedAllocator() = default;
    template<class U> constexpr ManagedAllocator(const ManagedAllocator<U>&) noexcept {}
    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        CUDA_CHECK(cudaMallocManaged(&ptr, n * sizeof(T)));
        return ptr;
    }
    void deallocate(T* p, std::size_t) noexcept {
        cudaFree(p);
    }
    template<class U> struct rebind { typedef ManagedAllocator<U> other; };
};

template<typename T>
using DeviceVector = std::vector<T, ManagedAllocator<T>>;

// CUDA Kernel Declarations
void cuda_matvec_add(const float* W, const float* x, float* out, int m, int n);
void cuda_matvec_sub(const float* W, const float* x, float* out, int m, int n);
void cuda_som_update(float* weights, const float* input, int bmu, float eff_lr, float r2, int n_neurons, int n_dims, int cols);
void cuda_som_distances(const float* weights, const float* input, float* dists, int n_neurons, int n_dims);
void cuda_device_synchronize();

} // namespace brain2

#else

// CPU Fallback Types
namespace brain2 {
template<typename T>
using DeviceVector = std::vector<T>;

inline void cuda_device_synchronize() {}
}

#endif
