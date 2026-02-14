//
// Copyright 2026 Rupert Nash, EPCC, University of Edinburgh
//
#include <algorithm>
#include <charconv>
#include <cmath>
#include <iostream>
#include <variant>
#include <vector>

#include "wave_cpu.h"
#include "wave_cuda.h"
#include "wave_omp.h"

namespace fs = std::filesystem;

using Clock = std::chrono::high_resolution_clock;
using Time = Clock::time_point;
using Dur = Clock::duration;

// For errors in parsing commandline so this can be caught separately
struct argparse_error : std::runtime_error {
    // Construct with a modern format string and args.
    template <typename... Args>
    explicit argparse_error(std::format_string<Args...> f, Args... args) : std::runtime_error(std::format(f, std::forward<Args>(args)...)) {
    }
};

struct ParseResult {
    Params p;
    fs::path outbase;
};
ParseResult parse_args(int argc, char const* argv[]) {
    // Defaults if not specified.
    shape_t shape{32, 32, 32};
    double dx = 10.0; // metres
    double dt = 0.002; // seconds
    int nsteps = 100;
    int out_period = 10;
    int nBoundaryLayers = 4;
    fs::path outbase = "test";

    // Handle command line
    int nArgSeen = 0;
    for (int i = 1; i < argc; ++i) {
        auto a = std::string_view(argv[i]);

        if (a.starts_with('-')) {
            auto flag = a.substr(1);
            // Has an associated value
            ++i;
            if (i >= argc)
                throw argparse_error("No value for option {}", flag);
            auto val = std::string_view(argv[i]);

            auto read_fl = [&val](char const* flg, auto& var) {
                auto [ptr, ec] = std::from_chars(val.begin(), val.end(), var);
                if (ec != std::errc())
                    throw argparse_error("Error converting --{} to number", flg);
                if (ptr != val.end())
                    throw argparse_error("Did not consume all of --{} when converting", flg);
            };

            if (flag == "shape") {
                int shape_i = 0;
                for (std::size_t comma = 0; comma < std::string_view::npos; ++shape_i) {
                    if (shape_i == 3)
                        throw argparse_error("Too many values in shape");

                    comma = val.find(',');
                    auto part = val.substr(0, std::min(comma, val.size()));
                    val = val.substr(comma+1);
                    auto [ptr, ec] = std::from_chars(part.begin(), part.end(), shape[shape_i]);
                    if (ec != std::errc())
                        throw argparse_error("Error converting --shape[{}] to number", shape_i);
                    if (ptr != part.end())
                        throw argparse_error("Did not consume all of --shape[{}] when converting", shape_i);
                }
                if (shape_i != 3)
                    throw argparse_error("Not enough values in shape!");
            } else if (flag == "dx") {
                read_fl("dx", dx);
            } else if (flag == "dt") {
                read_fl("dt", dt);
            } else if (flag == "nsteps") {
                read_fl("nsteps", nsteps);
            } else if (flag == "out_period") {
                read_fl("out_period", out_period);
            } else {
                throw argparse_error("Unknown flag {}", flag);
            }
        } else {
            if (nArgSeen)
                throw argparse_error("Unexpected positional argument {}", a);
            ++nArgSeen;
            outbase = a;
        }
    }
    return {
        Params{.dx=dx, .dt=dt, .shape=shape, .nsteps=nsteps, .out_period=out_period, .nBoundaryLayers=nBoundaryLayers},
        outbase
    };
}

int main(int argc, char const* argv[]) {
    try {
        // Parse command line args
        auto const [p, outbase] = parse_args(argc, argv);

        // Set the different output files:
        auto mkcp = [&outbase] (auto infix) -> fs::path {
            auto stem = outbase.stem();
            auto newstem = stem.string();
            newstem += infix;
            newstem += ".vtkhdf";
            if (outbase.has_parent_path())
                return outbase.parent_path() / newstem;
            else
                return newstem;
        };
        auto cpu_cp = mkcp(".cpu");
        auto cuda_cp = mkcp(".cuda");
        auto omp_cp = mkcp(".omp");

        // Initialise based on params on the CPU
        auto cpu_state = CpuWaveSimulation::from_params(cpu_cp, p);
        // Set up the GPU implementations as copies of this
        auto cuda_state = CudaWaveSimulation::from_cpu_sim(cuda_cp, cpu_state);
        auto omp_state = OmpWaveSimulation::from_cpu_sim(omp_cp, cpu_state);

        // Timing stats
        auto n_chunks = ceildiv(p.nsteps, p.out_period);
        auto cpu_time_s = std::vector<float>(n_chunks);
        auto cuda_time_s = std::vector<float>(n_chunks);
        auto omp_time_s = std::vector<float>(n_chunks);

        // Runner/benchmarker lambda
        auto benchmarker = [&](
                WaveSimulation& state
        ) {
            std::vector<float> time_s(n_chunks);
            std::vector<float> sups(n_chunks);
            out("Starting run with {}, timing in {} chunks", state.ID(), n_chunks);
            auto& t = state.u.time();
            auto [nx, ny, nz] = state.params.shape;
            auto const nsites = nx*ny*nz;
            for (int i = 0; i < n_chunks; ++i) {
                auto len = std::min(t + p.out_period, p.nsteps) - t;
                Time const start = Clock::now();
                state.run(len);
                Time const stop = Clock::now();
                std::chrono::duration<float> dt{stop - start};
                time_s[i] = dt.count();
                sups[i] = float(nsites * len) / dt.count();
                // Exclude IO from timings
                state.append_u_fields();
                out("Chunk {}, length = {}, time = {} s", i, len, dt.count());
            }
            return std::make_pair(time_s, sups);
        };
        // Lamdba to check the results of GPU versions match the CPU reference
        auto checker = [&cpu_state, eps=1e-8](WaveSimulation const& sim) {
            out("Checking {} results...", sim.ID());
            auto nerr = 0;
            auto const& ref_u = cpu_state.u.now();
            auto const& test_u = sim.u.now();
            auto shape = cpu_state.params.shape;
            auto padded_shape = shape;
            std::for_each(padded_shape.begin(), padded_shape.end(), [](auto& _) {_ += 2;});
            auto [L,M,N] = padded_shape;
            for (unsigned i = 0; i < L; ++i) {
                for (unsigned j = 0; j < M; ++j) {
                    for (unsigned k = 0; k < N; ++k) {
                        if (!approxEq(ref_u(i, j, k), test_u(i, j, k), eps)) {
                            nerr += 1;
                            out("Fields differ at ({}, {}, {})", i, j, k);
                        }
                    }
                }
            }
            out("Number of differences detected = {}", nerr);
            return nerr;
        };

        int total_errs = 0;
        auto cpu_stats = benchmarker(cpu_state);
        auto cuda_stats = benchmarker(cuda_state);
        total_errs += checker(cuda_state);
        auto omp_stats = benchmarker(omp_state);
        total_errs += checker(omp_state);

        auto print_stats = [] (auto const& stat_pair, char const* where) {
            std::vector<float> const& time = stat_pair.first;
            std::vector<float> const& sups = stat_pair.second;
            // Compute and print stats
            auto N = time.size();
            auto statter = [&N](std::vector<float> const& data) {
                auto min = std::numeric_limits<double>::infinity();
                double max = -min;
                double tsum = 0.0, tsumsq = 0.0;
                for (int i = 0; i < N; ++i) {
                    double const& t = data[i];
                    tsum += t;
                    tsumsq += t * t;
                    min = (t < min) ? t : min;
                    max = (t > max) ? t : max;
                }
                double mean = tsum / N;
                double tvar = (tsumsq - tsum * tsum / N) / (N - 1);
                double std = std::sqrt(tvar);
                out("min = {:.3e}, max = {:.3e}, mean = {:.3e}, std = {:.3e}",
                    min, max, mean, std);
            };

            out("Summary for {}", where);
            out("Run time / seconds");
            statter(time);
            out("Performance / (site updates per second)");
            statter(sups);
        };

        print_stats(cpu_stats, "CPU");
        print_stats(cuda_stats, "CUDA");
        print_stats(omp_stats, "OpenMP");

        return total_errs;
    } catch (argparse_error& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Usage:\n"
                     "awave [-shape int,int,int] [-dx float] [-dt float] \\\n"
                     "      [-nsteps int] [-out_period int] [output_base]\n"
                     "\n";
        return 1;

    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
