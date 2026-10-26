// Minimal CUDA allocation sanity check for very large problem sizes.
//
// Purpose:
//   The coursework driver (main.cpp) constructs both the CUDA and OpenMP GPU
//   simulation objects up-front. That means their device allocations must fit
//   *at the same time* on a single A100.
//
// This helper allocates the same order-of-magnitude buffers as the two GPU
// implementations combined, without running the full simulation or producing
// enormous HDF5 output files (which would be impractical for 1000^3 domains).
//
// Build (from repo root):
//   nvcc -O2 -std=c++17 -o build-dev/gpu_alloc_check build-dev/gpu_alloc_check.cu
//
// Run:
//   build-dev/gpu_alloc_check 1000 1000 1000
//
#include <cuda_runtime.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>

static void cuda_check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return;
    std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
    std::exit(2);
}

static void print_meminfo(char const* tag) {
    std::size_t free_b = 0, total_b = 0;
    cuda_check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    std::fprintf(stdout,
                 "%s: free=%0.2f GiB total=%0.2f GiB\n",
                 tag,
                 static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(total_b) / (1024.0 * 1024.0 * 1024.0));
}

int main(int argc, char** argv) {
    int nx = 1000, ny = 1000, nz = 1000;
    if (argc == 4) {
        nx = std::atoi(argv[1]);
        ny = std::atoi(argv[2]);
        nz = std::atoi(argv[3]);
    } else if (argc != 1) {
        std::fprintf(stderr, "Usage: %s [nx ny nz]\n", argv[0]);
        return 1;
    }

    // Match the coursework: u fields are padded by 1 halo in each direction.
    std::size_t const padded =
            static_cast<std::size_t>(nx + 2) *
            static_cast<std::size_t>(ny + 2) *
            static_cast<std::size_t>(nz + 2);
    std::size_t const u_bytes = padded * sizeof(double);

    // The optimised implementations store coefficients in compressed form:
    //   cs2_k: nz doubles
    //   damp_xy: nx*ny doubles
    std::size_t const cs2_bytes = static_cast<std::size_t>(nz) * sizeof(double);
    std::size_t const damp_bytes = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * sizeof(double);

    // In the driver program both GPU implementations exist concurrently.
    // With the 2-buffer scheme each implementation allocates:
    //   2 * u_bytes  + cs2_bytes + damp_bytes
    // so combined CUDA+OpenMP allocations are doubled.
    std::size_t const total_bytes =
            2 * (2 * u_bytes + cs2_bytes + damp_bytes);

    std::fprintf(stdout, "Requested shape: %d x %d x %d\n", nx, ny, nz);
    std::fprintf(stdout, "Padded elements: %" PRIu64 "\n", static_cast<std::uint64_t>(padded));
    std::fprintf(stdout, "Per-field bytes: %" PRIu64 " (%0.2f GiB)\n",
                 static_cast<std::uint64_t>(u_bytes),
                 static_cast<double>(u_bytes) / (1024.0 * 1024.0 * 1024.0));
    std::fprintf(stdout, "Total device bytes (CUDA+OpenMP, 2 buffers each): %" PRIu64 " (%0.2f GiB)\n",
                 static_cast<std::uint64_t>(total_bytes),
                 static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0));

    cuda_check(cudaFree(nullptr), "cudaFree(nullptr) init"); // init runtime early
    print_meminfo("Before allocations");

    void* ptrs[8] = {};
    int p = 0;

    // CUDA implementation: u_prev/u_now + compressed coeffs
    cuda_check(cudaMalloc(&ptrs[p++], u_bytes), "cudaMalloc cuda_u_prev");
    cuda_check(cudaMalloc(&ptrs[p++], u_bytes), "cudaMalloc cuda_u_now");
    cuda_check(cudaMalloc(&ptrs[p++], cs2_bytes), "cudaMalloc cuda_cs2_k");
    cuda_check(cudaMalloc(&ptrs[p++], damp_bytes), "cudaMalloc cuda_damp_xy");

    // OpenMP implementation: u_prev/u_now + compressed coeffs
    cuda_check(cudaMalloc(&ptrs[p++], u_bytes), "cudaMalloc omp_u_prev");
    cuda_check(cudaMalloc(&ptrs[p++], u_bytes), "cudaMalloc omp_u_now");
    cuda_check(cudaMalloc(&ptrs[p++], cs2_bytes), "cudaMalloc omp_cs2_k");
    cuda_check(cudaMalloc(&ptrs[p++], damp_bytes), "cudaMalloc omp_damp_xy");

    print_meminfo("After allocations");

    for (int i = 0; i < p; ++i) {
        cuda_check(cudaFree(ptrs[i]), "cudaFree");
    }
    print_meminfo("After free");

    std::fprintf(stdout, "OK: allocations succeeded.\n");
    return 0;
}
