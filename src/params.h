// -*- mode: C++; -*-
//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#ifndef AWAVE_PARAMS_H
#define AWAVE_PARAMS_H

#include <format>

#include "ndarray.h"

using array3d = nd::array<double, 3>;
using shape_t = array3d::index_type;

struct Params {
    double dx;
    double dt;
    shape_t shape;
    int nsteps;
    int out_period;
    int nBoundaryLayers;
};

// Allowing formatting of parameters
template<>
struct std::formatter<Params, char> {
    // Deal with the format string, don't allow anything for simplicity
    template<class Ctx>
    constexpr Ctx::iterator parse(Ctx& ctx) {
        auto it = ctx.begin();
        auto end = ctx.end();

        // Ensure no unexpected characters after the colon (if any)
        if (it != end && *it != '}') {
            throw std::format_error("Invalid format specifier for Params");
        }
        return it;
    }
    // Actually do the formatting
    template<class Ctx>
    Ctx::iterator format(Params const& p, Ctx& ctx) const {
        auto [nx, ny, nz] = p.shape;
        return std::format_to(ctx.out(), "Grid shape: [{}, {}, {}]\n"
                                         "Grid spacing: {} m\n"
                                         "Time step: {} s\n"
                                         "Number of steps: {}\n"
                                         "Output period: {}",
                              nx, ny, nz,  p.dx, p.dt, p.nsteps, p.out_period);
    }
};
#endif
