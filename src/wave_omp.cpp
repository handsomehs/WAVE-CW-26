//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "wave_omp.h"

#include <omp.h>

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
    int c_stride_y = 0;
    int c_stride_x = 0;
    std::size_t u_size = 0;
    std::size_t field_size = 0;

    double* d_prev = nullptr;
    double* d_now = nullptr;
    double* d_next = nullptr;
    double* d_cs2 = nullptr;
    double* d_damp = nullptr;

    OmpImplementationData(Params const& params, uField const& u, array3d const& cs2, array3d const& damp) {
        device = omp_get_default_device();
        auto shape = params.shape;
        nx = static_cast<int>(shape[0]);
        ny = static_cast<int>(shape[1]);
        nz = static_cast<int>(shape[2]);
        u_stride_y = nz + 2;
        u_stride_x = (ny + 2) * (nz + 2);
        c_stride_y = nz;
        c_stride_x = ny * nz;
        u_size = static_cast<std::size_t>(nx + 2) * static_cast<std::size_t>(ny + 2) * static_cast<std::size_t>(nz + 2);
        field_size = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);

        // Device-resident fields: keep on device for the whole simulation.
        d_prev = static_cast<double*>(omp_target_alloc(u_size * sizeof(double), device));
        d_now = static_cast<double*>(omp_target_alloc(u_size * sizeof(double), device));
        d_next = static_cast<double*>(omp_target_alloc(u_size * sizeof(double), device));
        d_cs2 = static_cast<double*>(omp_target_alloc(field_size * sizeof(double), device));
        d_damp = static_cast<double*>(omp_target_alloc(field_size * sizeof(double), device));

        auto host = omp_get_initial_device();
        // Initial H2D transfer for all inputs.
        omp_target_memcpy(d_prev, u.prev().data(), u_size * sizeof(double), 0, 0, device, host);
        omp_target_memcpy(d_now, u.now().data(), u_size * sizeof(double), 0, 0, device, host);
        omp_target_memcpy(d_next, u.next().data(), u_size * sizeof(double), 0, 0, device, host);
        omp_target_memcpy(d_cs2, cs2.data(), field_size * sizeof(double), 0, 0, device, host);
        omp_target_memcpy(d_damp, damp.data(), field_size * sizeof(double), 0, 0, device, host);
    }
    ~OmpImplementationData() {
        if (d_prev) omp_target_free(d_prev, device);
        if (d_now) omp_target_free(d_now, device);
        if (d_next) omp_target_free(d_next, device);
        if (d_cs2) omp_target_free(d_cs2, device);
        if (d_damp) omp_target_free(d_damp, device);
   }
};

OmpWaveSimulation::OmpWaveSimulation() = default;
OmpWaveSimulation::OmpWaveSimulation(OmpWaveSimulation&&)  noexcept = default;
OmpWaveSimulation& OmpWaveSimulation::operator=(OmpWaveSimulation&&) = default;
OmpWaveSimulation::~OmpWaveSimulation() = default;

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


void OmpWaveSimulation::append_u_fields() {
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

    for (int step = 0; step < n; ++step) {
        double* u_prev = impl.d_prev;
        double* u_now = impl.d_now;
        double* u_next = impl.d_next;
        double* d_cs2 = impl.d_cs2;
        double* d_damp = impl.d_damp;
        int u_stride_x = impl.u_stride_x;
        int u_stride_y = impl.u_stride_y;
        int c_stride_x = impl.c_stride_x;
        int c_stride_y = impl.c_stride_y;

        // The damping field is zero everywhere except in a small number of x/y
        // boundary layers (see initialisation in wave_cpu.cpp). Splitting the
        // domain lets the bulk update avoid loading `d_damp` and removes the
        // per-point branch.
        if (have_interior) {
            // 1) Interior (undamped): i=[nbl, nx-nbl), j=[nbl, ny-nbl).
            #pragma omp target teams distribute parallel for collapse(3) \
                device(device) \
                is_device_ptr(u_prev, u_now, u_next, d_cs2, d_damp)
            for (int ii = 0; ii < nx_inner; ++ii) {
                for (int jj = 0; jj < ny_inner; ++jj) {
                    for (int k = 0; k < nz; ++k) {
                        int i = ii + nbl;
                        int j = jj + nbl;
                        int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
                        int c_idx = i * c_stride_x + j * c_stride_y + k;

                        double center = u_now[u_idx];
                        double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
                                   + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
                                   + u_now[u_idx - 1] + u_now[u_idx + 1]
                                   - 6.0 * center;
                        double value = factor * d_cs2[c_idx] * lap;
                        u_next[u_idx] = 2.0 * center - u_prev[u_idx] + value;
                    }
                }
            }

            // 2) x boundary layers (includes corners). Combine low/high slabs
            // by mapping i_in in [0, 2*nbl) to global i.
            #pragma omp target teams distribute parallel for collapse(3) \
                device(device) \
                is_device_ptr(u_prev, u_now, u_next, d_cs2, d_damp)
            for (int i_in = 0; i_in < 2 * nbl; ++i_in) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        int i = i_in + ((i_in >= nbl) ? (nx - 2 * nbl) : 0);
                        int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
                        int c_idx = i * c_stride_x + j * c_stride_y + k;

                        double center = u_now[u_idx];
                        double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
                                   + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
                                   + u_now[u_idx - 1] + u_now[u_idx + 1]
                                   - 6.0 * center;
                        double value = factor * d_cs2[c_idx] * lap;

                        double d = d_damp[c_idx];
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
                is_device_ptr(u_prev, u_now, u_next, d_cs2, d_damp)
            for (int ii = 0; ii < nx_inner; ++ii) {
                for (int j_in = 0; j_in < 2 * nbl; ++j_in) {
                    for (int k = 0; k < nz; ++k) {
                        int i = ii + nbl;
                        int j = j_in + ((j_in >= nbl) ? (ny - 2 * nbl) : 0);
                        int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
                        int c_idx = i * c_stride_x + j * c_stride_y + k;

                        double center = u_now[u_idx];
                        double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
                                   + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
                                   + u_now[u_idx - 1] + u_now[u_idx + 1]
                                   - 6.0 * center;
                        double value = factor * d_cs2[c_idx] * lap;

                        double d = d_damp[c_idx];
                        double inv_den = 1.0 / (1.0 + d * dt);
                        double num = 1.0 - d * dt;
                        value *= inv_den;
                        u_next[u_idx] = 2.0 * inv_den * center - num * inv_den * u_prev[u_idx] + value;
                    }
                }
            }
        } else {
            // Fallback for very small problems: one kernel with the original
            // per-point damping branch.
            #pragma omp target teams distribute parallel for collapse(3) \
                device(device) \
                is_device_ptr(u_prev, u_now, u_next, d_cs2, d_damp)
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        int u_idx = (i + 1) * u_stride_x + (j + 1) * u_stride_y + (k + 1);
                        int c_idx = i * c_stride_x + j * c_stride_y + k;

                        double center = u_now[u_idx];
                        double lap = u_now[u_idx - u_stride_x] + u_now[u_idx + u_stride_x]
                                   + u_now[u_idx - u_stride_y] + u_now[u_idx + u_stride_y]
                                   + u_now[u_idx - 1] + u_now[u_idx + 1]
                                   - 6.0 * center;
                        double value = factor * d_cs2[c_idx] * lap;

                        double d = d_damp[c_idx];
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
        auto* old_prev = impl.d_prev;
        impl.d_prev = impl.d_now;
        impl.d_now = impl.d_next;
        impl.d_next = old_prev;
        u.advance();
    }

    // Copy back only after this run chunk for output and validation.
    auto host = omp_get_initial_device();
    omp_target_memcpy(u.now().data(), impl.d_now, impl.u_size * sizeof(double), 0, 0, host, impl.device);
    omp_target_memcpy(u.prev().data(), impl.d_prev, impl.u_size * sizeof(double), 0, 0, host, impl.device);
}
