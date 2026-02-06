cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(ASPP)

# We need this to use OpenMP offloading
set(CMAKE_CXX_COMPILER nvc++)

project(awave LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)

get_nvhpc_flags()

# NVC++ likes to target the host ISA very specifically, which is Intel
# for ASPP VM but AMD for GPU nodes
set(CMAKE_CXX_FLAGS -march=x86-64-v3)
# Be relaxed about constexpr device functions and use host-side compiler
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr -ccbin ${CMAKE_CXX_COMPILER} -march=x86-64-v3")
# Use nvc++ to link but enable CUDA: easiest wave to mix OpenMP and CUDA
set(CMAKE_EXE_LINKER_FLAGS -cuda)

# Set OpenMP to GPU mode
find_package(OpenMP REQUIRED)
set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
  INTERFACE_COMPILE_OPTIONS "-mp=gpu;${NVHPC_GPU_FLAGS}"
  INTERFACE_LINK_OPTIONS "-mp=gpu")

# Use HDF for IO
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_executable(awave
        h5io.cpp
        init_sos.cpp
        ufield.cpp
        wave_cpu.cpp
        wave_cuda.cu
	wave_omp.cpp
        main.cpp
)

target_link_libraries(awave hdf5::hdf5 OpenMP::OpenMP_CXX)
