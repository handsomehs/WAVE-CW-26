// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "wave_cuda.h"

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <utility>

// Free helper macro to check for CUDA errors!
#define CUDA_CHECK(expr) do { \
    cudaError_t res = expr; \
    if (res != cudaSuccess) \
      throw std::runtime_error(std::format(__FILE__ ":{} CUDA error: {}", __LINE__, cudaGetErrorString(res))); \
  } while (0)

static std::size_t padded_size(Params const& params) {
    auto [nx, ny, nz] = params.shape;
    return static_cast<std::size_t>(nx + 2) * static_cast<std::size_t>(ny + 2) * static_cast<std::size_t>(nz + 2);
}

static std::size_t field_size(Params const& params) {
    auto [nx, ny, nz] = params.shape;
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
}

// This struct can hold any data you need to manage running on the device
//
// Allocate with std::make_unique when you create the simulation
// object below, in `from_cpu_sim`.
struct CudaImplementationData {
    int device = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int u_stride_y = 0;
    int u_stride_x = 0;
    int c_stride_y = 0;
    int c_stride_x = 0;
    std::size_t u_size = 0;
    std::size_t field_size = 0;

    double* d_prev = nullptr;
    double* d_now = nullptr;
    double* d_next = nullptr;
    double* d_cs2 = nullptr;
    double* d_damp = nullptr;

    cudaStream_t stream = nullptr;

    CudaImplementationData(Params const& params, uField const& u, array3d const& cs2, array3d const& damp) {
        nvtx3::scoped_range r{"initialise"};
        CUDA_CHECK(cudaGetDevice(&device));
        auto shape = params.shape;
        nx = static_cast<int>(shape[0]);
        ny = static_cast<int>(shape[1]);
        nz = static_cast<int>(shape[2]);
        u_stride_y = nz + 2;
        u_stride_x = (ny + 2) * (nz + 2);
        c_stride_y = nz;
        c_stride_x = ny * nz;
        u_size = padded_size(params);
        field_size = ::field_size(params);

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // Device-resident fields: keep on GPU for the whole simulation.
        CUDA_CHECK(cudaMalloc(&d_prev, u_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_now, u_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_next, u_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cs2, field_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_damp, field_size * sizeof(double)));

        // Initial H2D transfer for all inputs.
        CUDA_CHECK(cudaMemcpyAsync(d_prev, u.prev().data(), u_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_now, u.now().data(), u_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_next, u.next().data(), u_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_cs2, cs2.data(), field_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_damp, damp.data(), field_size * sizeof(double), cudaMemcpyHostToDevice, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    ~CudaImplementationData() {
        if (d_prev) cudaFree(d_prev);
        if (d_now) cudaFree(d_now);
        if (d_next) cudaFree(d_next);
        if (d_cs2) cudaFree(d_cs2);
        if (d_damp) cudaFree(d_damp);
        if (stream) cudaStreamDestroy(stream);
   }
};

CudaWaveSimulation::CudaWaveSimulation() = default;
CudaWaveSimulation::CudaWaveSimulation(CudaWaveSimulation&&) noexcept = default;
CudaWaveSimulation& CudaWaveSimulation::operator=(CudaWaveSimulation&&) noexcept = default;
CudaWaveSimulation::~CudaWaveSimulation() = default;

CudaWaveSimulation CudaWaveSimulation::from_cpu_sim(const fs::path& cp, const WaveSimulation& source) {
    CudaWaveSimulation ans;
    out("Initialising {} simulation as copy of {}...", ans.ID(), source.ID());
    ans.params = source.params;
    ans.u = source.u.clone();
    ans.sos = source.sos.clone();
    ans.cs2 = source.cs2.clone();
    ans.damp = source.damp.clone();

    ans.checkpoint = cp;
    ans.h5 = H5IO::from_params(cp, ans.params);

    out("Write initial conditions to {}", ans.checkpoint.c_str());
    ans.h5.put_params(ans.params);
    ans.h5.put_damp(ans.damp);
    ans.h5.put_sos(ans.sos);
    ans.append_u_fields();

    int count = 0;
    auto err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        // Fallback keeps correctness on CPU-only systems.
        out("No CUDA device found; CUDA simulation will run on CPU");
        return ans;
    }

    ans.impl = std::make_unique<CudaImplementationData>(ans.params, ans.u, ans.cs2, ans.damp);

    return ans;
}

void CudaWaveSimulation::append_u_fields() {
    h5.append_u(u);
}

static void step_cpu(Params const& params, array3d const& cs2, array3d const& damp, uField& u) {
    auto d2 = params.dx * params.dx;
    auto dt = params.dt;
    auto factor = dt*dt / d2;
    auto [nx, ny, nz] = params.shape;
    for (unsigned i = 0; i < nx; ++i) {
        auto ii = i + 1;
        for (unsigned j = 0; j < ny; ++j) {
            auto jj = j + 1;
            for (unsigned k = 0; k < nz; ++k) {
                auto kk = k + 1;
                // Simple approximation of Laplacian
                auto value = factor * cs2(i, j, k) * (
                        u.now()(ii - 1, jj, kk) + u.now()(ii + 1, jj, kk) +
                        u.now()(ii, jj - 1, kk) + u.now()(ii, jj + 1, kk) +
                        u.now()(ii, jj, kk - 1) + u.now()(ii, jj, kk + 1)
                        - 6.0 * u.now()(ii, jj, kk)
                );
                // Deal with the damping field
                auto& d = damp(i, j, k);
                if (d == 0.0) {
                    u.next()(ii, jj, kk) = 2.0 * u.now()(ii, jj, kk) - u.prev()(ii, jj, kk) + value;
                } else {
                    auto inv_denominator = 1.0 / (1.0 + d * dt);
                    auto numerator = 1.0 - d * dt;
                    value *= inv_denominator;
                    u.next()(ii, jj, kk) = 2.0 * inv_denominator * u.now()(ii, jj, kk) -
                                           numerator * inv_denominator * u.prev()(ii, jj, kk) + value;
                }
            }
        }
    }
    u.advance();
}

__global__ void step_kernel(double const* __restrict__ u_prev,
                            double const* __restrict__ u_now,
                            double* __restrict__ u_next,
                            double const* __restrict__ cs2,
                            double const* __restrict__ damp,
                            int nx, int ny, int nz,
                            int u_stride_x, int u_stride_y,
                            int c_stride_x, int c_stride_y,
                            double factor, double dt) {
    int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (i >= nx || j >= ny || k >= nz) return;

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
    int c_idx = i * c_stride_x + j * c_stride_y + k;

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2[c_idx] * lap;

    double d = damp[c_idx];
    if (d == 0.0) {
        u_next[u_idx] = 2.0 * center - u_prev[u_idx] + value;
    } else {
        double inv_den = 1.0 / (1.0 + d * dt);
        double num = 1.0 - d * dt;
        value *= inv_den;
        u_next[u_idx] = 2.0 * inv_den * center - num * inv_den * u_prev[u_idx] + value;
    }
}

void CudaWaveSimulation::run(int n) {
    nvtx3::scoped_range r{"run"};
    if (!impl) {
        for (int i = 0; i < n; ++i) {
            step_cpu(params, cs2, damp, u);
        }
        return;
    }

    auto& impl = *this->impl;
    auto [nx, ny, nz] = params.shape;
    double dt = params.dt;
    double factor = dt * dt / (params.dx * params.dx);

    dim3 block(32, 4, 2);
    dim3 grid(ceildiv(static_cast<int>(nz), static_cast<int>(block.x)),
              ceildiv(static_cast<int>(ny), static_cast<int>(block.y)),
              ceildiv(static_cast<int>(nx), static_cast<int>(block.z)));

    for (int i = 0; i < n; ++i) {
        step_kernel<<<grid, block, 0, impl.stream>>>(
                impl.d_prev, impl.d_now, impl.d_next,
                impl.d_cs2, impl.d_damp,
                static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz),
                impl.u_stride_x, impl.u_stride_y,
                impl.c_stride_x, impl.c_stride_y,
                factor, dt);
        CUDA_CHECK(cudaGetLastError());

        // Rotate device pointers; avoids device memcpy each step.
        auto* old_prev = impl.d_prev;
        impl.d_prev = impl.d_now;
        impl.d_now = impl.d_next;
        impl.d_next = old_prev;
        u.advance();
    }

    // Copy back only after this run chunk for output and validation.
    CUDA_CHECK(cudaMemcpyAsync(u.now().data(), impl.d_now, impl.u_size * sizeof(double),
                               cudaMemcpyDeviceToHost, impl.stream));
    CUDA_CHECK(cudaMemcpyAsync(u.prev().data(), impl.d_prev, impl.u_size * sizeof(double),
                               cudaMemcpyDeviceToHost, impl.stream));
    CUDA_CHECK(cudaStreamSynchronize(impl.stream));
}
