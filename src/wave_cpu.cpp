//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include "wave_cpu.h"

#include <algorithm>
#include <iostream>

#include "init_sos.h"

//// Restart from a checkpoint
//CpuWaveSimulation CpuWaveSimulation::from_file(const fs::path& cp) {
//    CpuWaveSimulation self;
//    self.checkpoint = cp;
//    self.h5 = H5IO::from_file(self.checkpoint);
//    self.params = self.h5.get_params();
//    self.damp = self.h5.get_damp();
//    self.sos = self.h5.get_sos();
//    self.cs2 = array3d(self.sos.shape());
//    for (std::size_t i = 0; i < self.sos.size(); ++i) {
//        auto& c = self.sos.data()[i];
//        self.cs2.data()[i] = c*c;
//    }
//
//    self.u = self.h5.get_last_u();
//    return self;
//}

CpuWaveSimulation CpuWaveSimulation::from_params(fs::path const& cp, Params const& p) {
    CpuWaveSimulation ans;
    out("Initialising {} simulation from parameters...\n{}", ans.ID(), p);
    ans.checkpoint = cp;
    ans.params = p;
    out("Initial pressure field is pulse in middle of domain");
    ans.u = uField([&]{
        auto padded_shape = ans.params.shape;
        std::for_each(padded_shape.begin(), padded_shape.end(), [](auto& _) {_ += 2;});
        auto u_init = array3d(padded_shape);
        std::fill_n(u_init.data(), u_init.size(), 0.0);
        auto [nx, ny, nz] = padded_shape;
        auto hx = nx / 2;
        auto hy = ny / 2;
        auto hz = nz / 2;
        auto val = 50.0;
        u_init(hx, hy, hz) = val;
        val /= -6.0;
        u_init(hx - 1, hy, hz) = val;
        u_init(hx + 1, hy, hz) = val;
        u_init(hx, hy - 1, hz) = val;
        u_init(hx, hy + 1, hz) = val;
        u_init(hx, hy, hz - 1) = val;
        u_init(hx, hy, hz + 1) = val;
        return u_init;
    }());

    auto [nx, ny, nz] = ans.params.shape;
    // Speed of sound
    out("Speed of sound is simple ocean depth model");
    ans.sos = array3d(ans.params.shape);
    ans.cs2 = array3d(ans.params.shape);
    for (unsigned k = 0; k < nz; ++k) {
        auto depth = k * ans.params.dx;
        auto cs = SpeedOfSoundProfile::convert(depth);
        for (unsigned i = 0; i < nx; ++i) {
            for (unsigned j = 0; j < ny; ++j) {
                ans.sos(i, j, k) = cs;
                ans.cs2(i, j, k) = cs * cs;
            }
        }
    }

    out("Damping field to avoid reflections in x & y directions (large quiet ocean)");
    ans.damp = array3d(ans.params.shape);
    // Zero in the bulk
    std::fill_n(ans.damp.data(), ans.damp.size(), 0.0);
    auto nbl = ans.params.nBoundaryLayers;
    std::vector<double> ramp;
    ramp.reserve(nbl);
    for (int i = 0; i < nbl; ++i) {
        double r = nbl - i;
        ramp.push_back(9 * SpeedOfSoundProfile::MAX() * r * r / (2 * nbl * nbl * nbl * ans.params.dx));
    }
    // Damp only in x and y directions (silent, infinite ocean...)
    // Prep y cos multiply below for corners
    for (unsigned i = 0; i < nx; ++i) {
        for (unsigned j = 0; j < nbl; ++j)
            std::fill_n(&ans.damp(i, j, 0U), nz, 1.0);
        for (unsigned j = ny-nbl; j < ny; ++j)
            std::fill_n(&ans.damp(i, j, 0U), nz, 1.0);
    }
    // Set x
    for (unsigned i = 0; i < nbl; ++i)
        std::fill_n(&ans.damp(i, 0u, 0u), ny * nz, ramp[i]);
    for (unsigned i = nx-nbl; i < nx; ++i)
        std::fill_n(&ans.damp(i, 0u, 0u), ny * nz, ramp[nx - i - 1]);

    // y - multiplying hence set to 1 above
    for (unsigned i = 0; i < nx; ++i) {
        auto y_slice_scaler = [&](unsigned j, double r) {
            double v = ans.damp(i, j, 0u) * r;
            std::fill_n(&ans.damp(i, j, 0u), nz, v);
        };
        for (unsigned j = 0; j < nbl; ++j)
            y_slice_scaler(j, ramp[j]);
        for (unsigned j = ny - nbl; j < ny; ++j)
            y_slice_scaler(j, ramp[ny - j - 1]);
    }

    out("Write initial conditions to {}", ans.checkpoint.c_str());
    ans.h5 = H5IO::from_params(ans.checkpoint, ans.params);
    ans.h5.put_params(ans.params);
    ans.h5.put_damp(ans.damp);
    ans.h5.put_sos(ans.sos);
    ans.append_u_fields();
    return ans;
}

void CpuWaveSimulation::append_u_fields() {
    h5.append_u(u);
}

static void step(Params const& params, array3d const& cs2, array3d const& damp, uField& u) {
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

void CpuWaveSimulation::run(int n) {
    for (int i = 0; i < n; ++i) {
        step(params, cs2, damp, u);
    }
}
