// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "wave_cuda.h"

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <cstdlib>
#include <utility>
#include <vector>

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

// Allow benchmarking different kernel decomposition strategies without
// recompiling. This is intentionally a simple numeric interface so it can be
// set from KGPU YAMLs:
//   AWAVE_KERNEL_MODE=1  -> force single-kernel update (branch on damping)
//   AWAVE_KERNEL_MODE=2  -> force two-kernel update (interior + combined boundary)
//   AWAVE_KERNEL_MODE=3  -> force three-kernel update (interior + x/y boundary kernels)
// Unset / any other value keeps "auto" behaviour (currently mode 1).
static int kernel_mode_from_env() {
    const char* env = std::getenv("AWAVE_KERNEL_MODE");
    if (!env || env[0] == '\0') return 0;
    if (env[0] == '1') return 1;
    if (env[0] == '2') return 2;
    if (env[0] == '3') return 3;
    return 0;
}

// Trigger the first kernel launch path early so one-time CUDA initialisation
// (module loading, runtime setup) does not distort the first timed chunk.
__global__ void warmup_kernel(double* __restrict__ u_now) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        u_now[0] = u_now[0];
    }
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
    std::size_t u_size = 0;

    double* d_prev = nullptr;
    double* d_now = nullptr;
    double* d_next = nullptr;
    // `cs2(i,j,k)` depends only on k, and `damp(i,j,k)` depends only on (i,j).
    // Storing these compressed reduces device memory traffic.
    double* d_cs2_k = nullptr;    // size nz
    double* d_damp_xy = nullptr;  // size nx*ny

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
        u_size = padded_size(params);

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // Device-resident fields: keep on GPU for the whole simulation.
        CUDA_CHECK(cudaMalloc(&d_prev, u_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_now, u_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_next, u_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cs2_k, static_cast<std::size_t>(nz) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_damp_xy, static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * sizeof(double)));

        // Initial H2D transfer for all inputs.
        CUDA_CHECK(cudaMemcpyAsync(d_prev, u.prev().data(), u_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_now, u.now().data(), u_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_next, u.next().data(), u_size * sizeof(double), cudaMemcpyHostToDevice, stream));
        // Build compressed coefficient arrays on host, then transfer once.
        std::vector<double> cs2_k(static_cast<std::size_t>(nz));
        for (int k = 0; k < nz; ++k) {
            cs2_k[static_cast<std::size_t>(k)] = cs2(0U, 0U, static_cast<unsigned>(k));
        }
        std::vector<double> damp_xy(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                damp_xy[static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(j)] =
                        damp(static_cast<unsigned>(i), static_cast<unsigned>(j), 0U);
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(d_cs2_k, cs2_k.data(), static_cast<std::size_t>(nz) * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_damp_xy, damp_xy.data(),
                                   static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Warm up the kernel launch path so the first timed chunk in `run()` is
        // not affected by one-time overheads.
        warmup_kernel<<<1, 1, 0, stream>>>(d_now);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    ~CudaImplementationData() {
        if (d_prev) cudaFree(d_prev);
        if (d_now) cudaFree(d_now);
        if (d_next) cudaFree(d_next);
        if (d_cs2_k) cudaFree(d_cs2_k);
        if (d_damp_xy) cudaFree(d_damp_xy);
        if (stream) cudaStreamDestroy(stream);
   }
};

CudaWaveSimulation::CudaWaveSimulation() = default;
CudaWaveSimulation::CudaWaveSimulation(CudaWaveSimulation&&) noexcept = default;
CudaWaveSimulation& CudaWaveSimulation::operator=(CudaWaveSimulation&&) noexcept = default;
CudaWaveSimulation::~CudaWaveSimulation() = default;

// Factory method to create a CUDA-accelerated simulation from a CPU-resident one.
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

// Copy device-resident u fields back to host and append to the checkpoint file.
void CudaWaveSimulation::append_u_fields() {
    if (impl) {
        // Ensure host copies of u are up to date before writing the checkpoint.
        // This keeps device-to-host transfers out of the timed `run()` region.
        nvtx3::scoped_range r{"copyback"};
        auto& impl = *this->impl;
        CUDA_CHECK(cudaMemcpyAsync(u.now().data(), impl.d_now, impl.u_size * sizeof(double),
                                   cudaMemcpyDeviceToHost, impl.stream));
        CUDA_CHECK(cudaMemcpyAsync(u.prev().data(), impl.d_prev, impl.u_size * sizeof(double),
                                   cudaMemcpyDeviceToHost, impl.stream));
        CUDA_CHECK(cudaStreamSynchronize(impl.stream));
    }
    h5.append_u(u);
}

// Naive CPU implementation for correctness checking and fallback when no CUDA device is available
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

// one-kernel update: each thread computes one point, 
// branching on the damping field to apply the correct update.
__global__ void step_kernel(double const* __restrict__ u_prev,
                            double const* __restrict__ u_now,
                            double* __restrict__ u_next,
                            double const* __restrict__ cs2_k,
                            double const* __restrict__ damp_xy,
                            int nx, int ny, int nz,
                            int u_stride_x, int u_stride_y,
                            double factor, double dt) {
    int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int i = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (i >= nx || j >= ny || k >= nz) return;

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2_k[k] * lap;

    double d = damp_xy[i * ny + j];
    if (d == 0.0) {
        u_next[u_idx] = 2.0 * center - u_prev[u_idx] + value;
    } else {
        double inv_den = 1.0 / (1.0 + d * dt);
        double num = 1.0 - d * dt;
        value *= inv_den;
        u_next[u_idx] = 2.0 * inv_den * center - num * inv_den * u_prev[u_idx] + value;
    }
}

// Interior update (no damping): the damping field is identically zero away from the
// x/y boundary layers, so we can avoid reading it and remove the branch entirely.
__global__ void step_kernel_interior(double const* __restrict__ u_prev,
                                     double const* __restrict__ u_now,
                                     double* __restrict__ u_next,
                                     double const* __restrict__ cs2_k,
                                     int nx_inner, int ny_inner, int nz,
                                     int u_stride_x, int u_stride_y,
                                     int nbl,
                                     double factor) {
    int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j_inner = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int i_inner = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (i_inner >= nx_inner || j_inner >= ny_inner || k >= nz) return;

    int i = i_inner + nbl;
    int j = j_inner + nbl;

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2_k[k] * lap;
    u_next[u_idx] = 2.0 * center - u_prev[u_idx] + value;
}

// Boundary update (with damping) on x boundary layers. We combine the low and high
// x-slabs into one launch by mapping i_in in [0, 2*nbl) to:
//   i = i_in                    for i_in < nbl
//   i = (nx - nbl) + (i_in-nbl) for i_in >= nbl
// This covers all x-boundary points (including corners), so the y-boundary kernel
// below must exclude x-boundary i values to avoid double writes.
__global__ void step_kernel_xbound(double const* __restrict__ u_prev,
                                   double const* __restrict__ u_now,
                                   double* __restrict__ u_next,
                                   double const* __restrict__ cs2_k,
                                   double const* __restrict__ damp_xy,
                                   int nx, int ny, int nz,
                                   int u_stride_x, int u_stride_y,
                                   int nbl,
                                   double factor, double dt) {
    int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int i_in = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (i_in >= 2 * nbl || j >= ny || k >= nz) return;

    int i = i_in + ((i_in >= nbl) ? (nx - 2 * nbl) : 0);

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2_k[k] * lap;

    double d = damp_xy[i * ny + j];
    double inv_den = 1.0 / (1.0 + d * dt);
    double num = 1.0 - d * dt;
    value *= inv_den;
    u_next[u_idx] = 2.0 * inv_den * center - num * inv_den * u_prev[u_idx] + value;
}

// Boundary update (with damping) on y boundary layers, excluding x-boundary slabs.
// We combine the low and high y-slabs in the same way as the x-boundary kernel.
__global__ void step_kernel_ybound(double const* __restrict__ u_prev,
                                   double const* __restrict__ u_now,
                                   double* __restrict__ u_next,
                                   double const* __restrict__ cs2_k,
                                   double const* __restrict__ damp_xy,
                                   int nx_inner, int ny, int nz,
                                   int u_stride_x, int u_stride_y,
                                   int nbl,
                                   double factor, double dt) {
    int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j_in = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int i_inner = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (i_inner >= nx_inner || j_in >= 2 * nbl || k >= nz) return;

    int i = i_inner + nbl;
    int j = j_in + ((j_in >= nbl) ? (ny - 2 * nbl) : 0);

    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2_k[k] * lap;

    double d = damp_xy[i * ny + j];
    double inv_den = 1.0 / (1.0 + d * dt);
    double num = 1.0 - d * dt;
    value *= inv_den;
    u_next[u_idx] = 2.0 * inv_den * center - num * inv_den * u_prev[u_idx] + value;
}

// Two-kernel update strategy: merge the interior kernel and the y-boundary
// kernel into one launch, and keep the x-boundary launch separate.
//
// Rationale: the damping field is non-zero iff (i in x-boundary) OR (j in
// y-boundary). This kernel runs only for i in the interior range, so it only
// needs to apply damping for y-boundary j values.
__global__ void step_kernel_core_ybound(double const* __restrict__ u_prev,
                                        double const* __restrict__ u_now,
                                        double* __restrict__ u_next,
                                        double const* __restrict__ cs2_k,
                                        double const* __restrict__ damp_xy,
                                        int nx_inner, int ny, int nz,
                                        int u_stride_x, int u_stride_y,
                                        int nbl,
                                        double factor, double dt) {
    int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    int i_inner = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
    if (i_inner >= nx_inner || j >= ny || k >= nz) return;

    int i = i_inner + nbl;
    int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);

    double center = u_now[u_idx];
    double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
               + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
               + u_now[u_idx - 1] + u_now[u_idx + 1]
               - 6.0 * center;
    double value = factor * cs2_k[k] * lap;

    bool const is_yb = (j < nbl) || (j >= (ny - nbl));
    if (!is_yb) {
        u_next[u_idx] = 2.0 * center - u_prev[u_idx] + value;
        return;
    }

    double d = damp_xy[i * ny + j];
    double inv_den = 1.0 / (1.0 + d * dt);
    double num = 1.0 - d * dt;
    value *= inv_den;
    u_next[u_idx] = 2.0 * inv_den * center - num * inv_den * u_prev[u_idx] + value;
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
    int const nx = impl.nx;
    int const ny = impl.ny;
    int const nz = impl.nz;
    int const nbl = params.nBoundaryLayers;
    double dt = params.dt;
    double factor = dt * dt / (params.dx * params.dx);

    dim3 block_ijk(32, 4, 2);
    dim3 grid_full(ceildiv(nz, static_cast<int>(block_ijk.x)),
                   ceildiv(ny, static_cast<int>(block_ijk.y)),
                   ceildiv(nx, static_cast<int>(block_ijk.z)));

    bool const have_interior = (nbl > 0) && (nx > 2 * nbl) && (ny > 2 * nbl);
    int const nx_inner = have_interior ? (nx - 2 * nbl) : 0;
    int const ny_inner = have_interior ? (ny - 2 * nbl) : 0;

    // Domain splitting reduces memory traffic in large problems, but increases
    // per-step launch overhead. We support multiple strategies:
    //   1 kernel: full domain, branch on damping (best for very small domains)
    //   2 kernel: (interior + y-boundary) + x-boundary kernels
    //   3 kernel: interior + x/y boundary kernels (most specialised)
    //
    // "Auto" defaults to the 1-kernel path; 2/3-kernel modes are still
    // available for benchmarking via AWAVE_KERNEL_MODE. The sweep on A100
    // (32..1000) shows mode 1 wins almost everywhere, so this is the safest
    // default for marking.
    int mode = 0; // 0=auto, 1=one-kernel, 2=two-kernel, 3=three-kernel
    int const requested_mode = kernel_mode_from_env();
    if (!have_interior) {
        mode = 1;
    } else if (requested_mode == 1 || requested_mode == 2 || requested_mode == 3) {
        mode = requested_mode;
    } else {
        mode = 1;
    }

    dim3 grid_inner;
    dim3 grid_xb;
    dim3 grid_yb;
    dim3 grid_core;
    if (mode == 3) {
        grid_inner = dim3(ceildiv(nz, static_cast<int>(block_ijk.x)),
                          ceildiv(ny_inner, static_cast<int>(block_ijk.y)),
                          ceildiv(nx_inner, static_cast<int>(block_ijk.z)));
        grid_xb = dim3(ceildiv(nz, static_cast<int>(block_ijk.x)),
                       ceildiv(ny, static_cast<int>(block_ijk.y)),
                       ceildiv(2 * nbl, static_cast<int>(block_ijk.z)));
        grid_yb = dim3(ceildiv(nz, static_cast<int>(block_ijk.x)),
                       ceildiv(2 * nbl, static_cast<int>(block_ijk.y)),
                       ceildiv(nx_inner, static_cast<int>(block_ijk.z)));
    } else if (mode == 2) {
        grid_core = dim3(ceildiv(nz, static_cast<int>(block_ijk.x)),
                         ceildiv(ny, static_cast<int>(block_ijk.y)),
                         ceildiv(nx_inner, static_cast<int>(block_ijk.z)));
        grid_xb = dim3(ceildiv(nz, static_cast<int>(block_ijk.x)),
                       ceildiv(ny, static_cast<int>(block_ijk.y)),
                       ceildiv(2 * nbl, static_cast<int>(block_ijk.z)));
    }

    for (int i = 0; i < n; ++i) {
        if (mode == 3) {
            // 1) Interior (undamped) region.
            step_kernel_interior<<<grid_inner, block_ijk, 0, impl.stream>>>(
                    impl.d_prev, impl.d_now, impl.d_next,
                    impl.d_cs2_k,
                    nx_inner, ny_inner, nz,
                    impl.u_stride_x, impl.u_stride_y,
                    nbl,
                    factor);
            CUDA_CHECK(cudaGetLastError());

            // 2) x boundary layers (includes corners).
            step_kernel_xbound<<<grid_xb, block_ijk, 0, impl.stream>>>(
                    impl.d_prev, impl.d_now, impl.d_next,
                    impl.d_cs2_k, impl.d_damp_xy,
                    nx, ny, nz,
                    impl.u_stride_x, impl.u_stride_y,
                    nbl,
                    factor, dt);
            CUDA_CHECK(cudaGetLastError());

            // 3) y boundary layers excluding x boundary layers.
            step_kernel_ybound<<<grid_yb, block_ijk, 0, impl.stream>>>(
                    impl.d_prev, impl.d_now, impl.d_next,
                    impl.d_cs2_k, impl.d_damp_xy,
                    nx_inner, ny, nz,
                    impl.u_stride_x, impl.u_stride_y,
                    nbl,
                    factor, dt);
            CUDA_CHECK(cudaGetLastError());
        } else if (mode == 2) {
            // Two-kernel split: (interior + y-boundary) + x-boundary.
            step_kernel_core_ybound<<<grid_core, block_ijk, 0, impl.stream>>>(
                    impl.d_prev, impl.d_now, impl.d_next,
                    impl.d_cs2_k, impl.d_damp_xy,
                    nx_inner, ny, nz,
                    impl.u_stride_x, impl.u_stride_y,
                    nbl,
                    factor, dt);
            CUDA_CHECK(cudaGetLastError());

            step_kernel_xbound<<<grid_xb, block_ijk, 0, impl.stream>>>(
                    impl.d_prev, impl.d_now, impl.d_next,
                    impl.d_cs2_k, impl.d_damp_xy,
                    nx, ny, nz,
                    impl.u_stride_x, impl.u_stride_y,
                    nbl,
                    factor, dt);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // Small-problem path: one kernel with the original per-point damping
            // branch to reduce launch overhead.
            step_kernel<<<grid_full, block_ijk, 0, impl.stream>>>(
                    impl.d_prev, impl.d_now, impl.d_next,
                    impl.d_cs2_k, impl.d_damp_xy,
                    nx, ny, nz,
                    impl.u_stride_x, impl.u_stride_y,
                    factor, dt);
            CUDA_CHECK(cudaGetLastError());
        }

        // Rotate device pointers; avoids device memcpy each step.
        // Keep the host-side time counter in sync for output and checking.
        auto* old_prev = impl.d_prev;
        impl.d_prev = impl.d_now;
        impl.d_now = impl.d_next;
        impl.d_next = old_prev;
        u.advance();
    }

    // Synchronise so the caller's timing covers kernel execution.
    CUDA_CHECK(cudaStreamSynchronize(impl.stream));
}
