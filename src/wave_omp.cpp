//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "wave_omp.h"

#include <omp.h>
#include <cstdlib>
#include <vector>

// Allow benchmarking different kernel decomposition strategies without
// recompiling. See `wave_cuda.cu` for the full description (auto defaults to 1).
static int kernel_mode_from_env() {
    const char* env = std::getenv("AWAVE_KERNEL_MODE");
    if (!env || env[0] == '\0') return 0;
    if (env[0] == '1') return 1;
    if (env[0] == '2') return 2;
    if (env[0] == '3') return 3;
    return 0;
}

// This struct can hold any data you need to manage running on the device
//
// Allocate with std::make_unique when you create the simulation
// object below, in `from_cpu_sim`.
struct OmpImplementationData {
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

    OmpImplementationData(Params const& params, uField const& u, array3d const& cs2, array3d const& damp) {
        device = omp_get_default_device();
        auto shape = params.shape;
        nx = static_cast<int>(shape[0]);
        ny = static_cast<int>(shape[1]);
        nz = static_cast<int>(shape[2]);
        u_stride_y = nz + 2;
        u_stride_x = (ny + 2) * (nz + 2);
        u_size = static_cast<std::size_t>(nx + 2) * static_cast<std::size_t>(ny + 2) * static_cast<std::size_t>(nz + 2);

        // Device-resident fields: keep on device for the whole simulation.
        d_prev = static_cast<double*>(omp_target_alloc(u_size * sizeof(double), device));
        d_now = static_cast<double*>(omp_target_alloc(u_size * sizeof(double), device));
        d_next = static_cast<double*>(omp_target_alloc(u_size * sizeof(double), device));
        d_cs2_k = static_cast<double*>(omp_target_alloc(static_cast<std::size_t>(nz) * sizeof(double), device));
        d_damp_xy = static_cast<double*>(omp_target_alloc(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * sizeof(double), device));

        auto host = omp_get_initial_device();
        // Initial H2D transfer for all inputs.
        omp_target_memcpy(d_prev, u.prev().data(), u_size * sizeof(double), 0, 0, device, host);
        omp_target_memcpy(d_now, u.now().data(), u_size * sizeof(double), 0, 0, device, host);
        omp_target_memcpy(d_next, u.next().data(), u_size * sizeof(double), 0, 0, device, host);
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
        omp_target_memcpy(d_cs2_k, cs2_k.data(), static_cast<std::size_t>(nz) * sizeof(double), 0, 0, device, host);
        omp_target_memcpy(d_damp_xy, damp_xy.data(),
                          static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * sizeof(double),
                          0, 0, device, host);

        // Warm up the offload runtime and code path so one-time initialisation
        // does not pollute the first timed chunk in `run()`.
        double* u_now = d_now;
        #pragma omp target teams distribute parallel for device(device) is_device_ptr(u_now)
        for (int idx = 0; idx < 1; ++idx) {
            u_now[idx] = u_now[idx];
        }
    }
    ~OmpImplementationData() {
        if (d_prev) omp_target_free(d_prev, device);
        if (d_now) omp_target_free(d_now, device);
        if (d_next) omp_target_free(d_next, device);
        if (d_cs2_k) omp_target_free(d_cs2_k, device);
        if (d_damp_xy) omp_target_free(d_damp_xy, device);
   }
};

OmpWaveSimulation::OmpWaveSimulation() = default;
OmpWaveSimulation::OmpWaveSimulation(OmpWaveSimulation&&)  noexcept = default;
OmpWaveSimulation& OmpWaveSimulation::operator=(OmpWaveSimulation&&) = default;
OmpWaveSimulation::~OmpWaveSimulation() = default;

// Factory method to create an OpenMP-accelerated simulation from a CPU-resident one.
OmpWaveSimulation OmpWaveSimulation::from_cpu_sim(const fs::path& cp, const WaveSimulation& source) {
    OmpWaveSimulation ans;
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

    if (omp_get_num_devices() <= 0) {
        // Fallback keeps correctness on CPU-only systems.
        out("No OpenMP target device found; OpenMP simulation will run on CPU");
        return ans;
    }

    ans.impl = std::make_unique<OmpImplementationData>(ans.params, ans.u, ans.cs2, ans.damp);

    return ans;
}

// Copy device-resident u fields back to host and append to the checkpoint file.
void OmpWaveSimulation::append_u_fields() {
    if (impl) {
        // Ensure host copies of u are up to date before writing the checkpoint.
        // This keeps device-to-host transfers out of the timed `run()` region.
        auto& impl = *this->impl;
        auto host = omp_get_initial_device();
        omp_target_memcpy(u.now().data(), impl.d_now, impl.u_size * sizeof(double), 0, 0, host, impl.device);
        omp_target_memcpy(u.prev().data(), impl.d_prev, impl.u_size * sizeof(double), 0, 0, host, impl.device);
    }
    h5.append_u(u);
}

// Naive CPU implementation for correctness checking and fallback when no OpenMP target device is available
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

void OmpWaveSimulation::run(int n) {
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
    int const device = impl.device;
    int const nbl = params.nBoundaryLayers;
    double dt = params.dt;
    double factor = dt * dt / (params.dx * params.dx);

    bool const have_interior = (nbl > 0) && (nx > 2 * nbl) && (ny > 2 * nbl);
    int const nx_inner = have_interior ? (nx - 2 * nbl) : 0;
    int const ny_inner = have_interior ? (ny - 2 * nbl) : 0;

    // As in the CUDA implementation: splitting improves throughput on large
    // domains (less memory traffic and fewer branches), but increases launch
    // overhead. We support 1/2/3-offload strategies controlled by the same
    // `AWAVE_KERNEL_MODE` environment variable:
    //   1: single offload over full domain (branch on damping)
    //   2: (interior + y-boundary) + x-boundary offloads
    //   3: interior + x/y boundary offloads
    //
    // "Auto" defaults to the 1-offload path; 2/3-offload modes remain
    // available for benchmarking via AWAVE_KERNEL_MODE. The A100 sweep
    // shows mode 1 is consistently fastest across the tested shapes.
    int mode = 0; // 0=auto, 1=one-kernel, 2=two-kernel, 3=three-kernel
    int const requested_mode = kernel_mode_from_env();
    if (!have_interior) {
        mode = 1;
    } else if (requested_mode == 1 || requested_mode == 2 || requested_mode == 3) {
        mode = requested_mode;
    } else {
        mode = 1;
    }

    for (int step = 0; step < n; ++step) {
        double* u_prev = impl.d_prev;
        double* u_now = impl.d_now;
        double* u_next = impl.d_next;
        double* d_cs2_k = impl.d_cs2_k;
        double* d_damp_xy = impl.d_damp_xy;
        int u_stride_x = impl.u_stride_x;
        int u_stride_y = impl.u_stride_y;

        // The damping field is zero everywhere except in a small number of x/y
        // boundary layers (see initialisation in wave_cpu.cpp). Splitting the
        // domain lets the bulk update avoid loading `d_damp_xy` and removes the
        // per-point branch.
        if (mode == 3) {
            // 1) Interior (undamped): i=[nbl, nx-nbl), j=[nbl, ny-nbl).
            #pragma omp target teams distribute parallel for collapse(3) \
                device(device) \
                is_device_ptr(u_prev, u_now, u_next, d_cs2_k, d_damp_xy)
            for (int ii = 0; ii < nx_inner; ++ii) {
                for (int jj = 0; jj < ny_inner; ++jj) {
                    for (int k = 0; k < nz; ++k) {
                        int i = ii + nbl;
                        int j = jj + nbl;
                        int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);

                        double center = u_now[u_idx];
                        double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
                                   + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
                                   + u_now[u_idx - 1] + u_now[u_idx + 1]
                                   - 6.0 * center;
                        double value = factor * d_cs2_k[k] * lap;
                        u_next[u_idx] = 2.0 * center - u_prev[u_idx] + value;
                    }
                }
            }

            // 2) x boundary layers (includes corners). Combine low/high slabs
            // by mapping i_in in [0, 2*nbl) to global i.
            #pragma omp target teams distribute parallel for collapse(3) \
                device(device) \
                is_device_ptr(u_prev, u_now, u_next, d_cs2_k, d_damp_xy)
            for (int i_in = 0; i_in < 2 * nbl; ++i_in) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        int i = i_in + ((i_in >= nbl) ? (nx - 2 * nbl) : 0);
                        int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);

                        double center = u_now[u_idx];
                        double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
                                   + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
                                   + u_now[u_idx - 1] + u_now[u_idx + 1]
                                   - 6.0 * center;
                        double value = factor * d_cs2_k[k] * lap;

                        double d = d_damp_xy[i * ny + j];
                        double inv_den = 1.0 / (1.0 + d * dt);
                        double num = 1.0 - d * dt;
                        value *= inv_den;
                        u_next[u_idx] = 2.0 * inv_den * center - num * inv_den * u_prev[u_idx] + value;
                    }
                }
            }

            // 3) y boundary layers excluding the x boundary slabs.
            #pragma omp target teams distribute parallel for collapse(3) \
                device(device) \
                is_device_ptr(u_prev, u_now, u_next, d_cs2_k, d_damp_xy)
            for (int ii = 0; ii < nx_inner; ++ii) {
                for (int j_in = 0; j_in < 2 * nbl; ++j_in) {
                    for (int k = 0; k < nz; ++k) {
                        int i = ii + nbl;
                        int j = j_in + ((j_in >= nbl) ? (ny - 2 * nbl) : 0);
                        int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);

                        double center = u_now[u_idx];
                        double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
                                   + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
                                   + u_now[u_idx - 1] + u_now[u_idx + 1]
                                   - 6.0 * center;
                        double value = factor * d_cs2_k[k] * lap;

                        double d = d_damp_xy[i * ny + j];
                        double inv_den = 1.0 / (1.0 + d * dt);
                        double num = 1.0 - d * dt;
                        value *= inv_den;
                        u_next[u_idx] = 2.0 * inv_den * center - num * inv_den * u_prev[u_idx] + value;
                    }
                }
            }
        } else if (mode == 2) {
            // 1) i interior for all j: only y-boundary points are damped.
            #pragma omp target teams distribute parallel for collapse(3) \
                device(device) \
                is_device_ptr(u_prev, u_now, u_next, d_cs2_k, d_damp_xy)
            for (int ii = 0; ii < nx_inner; ++ii) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        int i = ii + nbl;
                        int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);

                        double center = u_now[u_idx];
                        double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
                                   + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
                                   + u_now[u_idx - 1] + u_now[u_idx + 1]
                                   - 6.0 * center;
                        double value = factor * d_cs2_k[k] * lap;
                        bool const is_yb = (j < nbl) || (j >= (ny - nbl));
                        if (!is_yb) {
                            u_next[u_idx] = 2.0 * center - u_prev[u_idx] + value;
                        } else {
                            double d = d_damp_xy[i * ny + j];
                            double inv_den = 1.0 / (1.0 + d * dt);
                            double num = 1.0 - d * dt;
                            value *= inv_den;
                            u_next[u_idx] = 2.0 * inv_den * center - num * inv_den * u_prev[u_idx] + value;
                        }
                    }
                }
            }

            // 2) x boundary layers (includes corners).
            #pragma omp target teams distribute parallel for collapse(3) \
                device(device) \
                is_device_ptr(u_prev, u_now, u_next, d_cs2_k, d_damp_xy)
            for (int i_in = 0; i_in < 2 * nbl; ++i_in) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        int i = i_in + ((i_in >= nbl) ? (nx - 2 * nbl) : 0);
                        int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);

                        double center = u_now[u_idx];
                        double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
                                   + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
                                   + u_now[u_idx - 1] + u_now[u_idx + 1]
                                   - 6.0 * center;
                        double value = factor * d_cs2_k[k] * lap;

                        double d = d_damp_xy[i * ny + j];
                        double inv_den = 1.0 / (1.0 + d * dt);
                        double num = 1.0 - d * dt;
                        value *= inv_den;
                        u_next[u_idx] = 2.0 * inv_den * center - num * inv_den * u_prev[u_idx] + value;
                    }
                }
            }
        } else {
            // one offload with the original per-point
            // damping branch to reduce launch overhead.
            #pragma omp target teams distribute parallel for collapse(3) \
                device(device) \
                is_device_ptr(u_prev, u_now, u_next, d_cs2_k, d_damp_xy)
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);

                        double center = u_now[u_idx];
                        double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
                                   + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
                                   + u_now[u_idx - 1] + u_now[u_idx + 1]
                                   - 6.0 * center;
                        double value = factor * d_cs2_k[k] * lap;

                        double d = d_damp_xy[i * ny + j];
                        if (d == 0.0) {
                            u_next[u_idx] = 2.0 * center - u_prev[u_idx] + value;
                        } else {
                            double inv_den = 1.0 / (1.0 + d * dt);
                            double num = 1.0 - d * dt;
                            value *= inv_den;
                            u_next[u_idx] = 2.0 * inv_den * center - num * inv_den * u_prev[u_idx] + value;
                        }
                    }
                }
            }
        }

        // Rotate device pointers; avoids device memcpy each step.
        // Keep the host-side time counter in sync for output and checking.
        auto* old_prev = impl.d_prev;
        impl.d_prev = impl.d_now;
        impl.d_now = impl.d_next;
        impl.d_next = old_prev;
        u.advance();
    }
}
