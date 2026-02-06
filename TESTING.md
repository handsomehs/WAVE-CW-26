# Testing workflow (CPU / CUDA / OpenMP offload)

`awave` always runs, in order:

1. CPU reference
2. CUDA implementation
3. OpenMP target offload implementation

and checks CUDA/OpenMP results against the CPU reference (tolerance `eps=1e-8`).
See `src/main.cpp`.

This file documents a repeatable testing workflow on the ASPP VM + EIDF KGPU.

## 0) One-time setup

- Work inside your shared PVC path so the same build is visible inside KGPU jobs.
- Load toolchain modules on the ASPP VM (example; adjust if your environment differs):

```bash
module load cmake/3.25.2 nvidia/nvhpc/24.5
# If CMake cannot find HDF5:
# module spider hdf5
# module load hdf5/<version>
```

## 1) Build (ASPP VM)

Release build (recommended for performance runs):

```bash
cmake -S src -B build-dev -DCMAKE_BUILD_TYPE=Release
cmake --build build-dev -j
```

Developer build (symbols, still fairly fast):

```bash
cmake -S src -B build-dev -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-dev -j
```

Sanity check:

```bash
./build-dev/awave -h
```

## 2) Local smoke run (no GPU)

Useful to quickly catch obvious issues (CLI parsing, output paths, etc.).
On non-GPU nodes, CUDA/OpenMP may fall back to CPU, so performance numbers are not meaningful.

```bash
./build-dev/awave -shape 32,32,32 -nsteps 10 -out_period 5 /tmp/awave-local
```

Success criteria:
- The program finishes.
- It prints `Number of differences detected = 0` for both CUDA and OpenMP checks.
- It produces `/tmp/awave-local.{cpu,cuda,omp}.vtkhdf`.

## 3) GPU correctness (fast) - MIG A100 slice

Use the provided job: `run-correct-mig.yml` (A100 MIG 1g.5gb).

```bash
kgpu create -f run-correct-mig.yml
kgpu jls -n 'awave-correct-mig-*'
kgpu logs -j <job-name>
```

Success criteria (in logs):
- `Checking CUDA results...` then `Number of differences detected = 0`
- `Checking OpenMP results...` then `Number of differences detected = 0`

If either count is non-zero (or the job fails), fix correctness before doing performance runs.

## 4) GPU performance (full A100)

Marking uses full A100s (`NVIDIA-A100-SXM4-40GB`), so use those for performance.
Example jobs provided:

- `run-perf-a100-128.yml` (shape `128^3`, `nsteps=20`, `out_period=10`)
- `run-perf-a100.yml` (shape `256^3`, `nsteps=4`, `out_period=2`)

Submit + view logs:

```bash
kgpu create -f run-perf-a100-128.yml
kgpu jls -n 'awave-perf-a100-128-*'
kgpu logs -j <job-name>
```

What to record (from logs):
- For CPU/CUDA/OpenMP, the `Performance / (site updates per second)` **mean** value.
- Shape, `nsteps`, `out_period`, GPU type, and date/time.

Notes:
- Timings exclude I/O (`append_u_fields()` happens after each timed chunk), but outputs are still written.
- Prefer multiple chunks (i.e. `nsteps` is a multiple of `out_period`) and enough work per chunk.
- Use an output base under `/tmp` (as in the provided YAMLs) to avoid slow filesystem effects.

## 5) Sweeping sizes for the report

For plots/tables, run a small matrix of problem sizes. Example:
- 32, 64: overhead/correctness
- 128, 192, 256: core performance range
- 384, 512: if memory/time allow

Convenience jobs:
- `run-sweep-a100.yml`: runs a small size sweep on full A100s and deletes `/tmp` outputs after each case.
- `profile-nsys-32.yml`: nsys profile for `32^3` to highlight small-problem launch/runtime overhead.

Guidelines:
- Keep `out_period` large enough to avoid excessive output volume.
- Increase `nsteps` for small shapes to reduce launch/overhead noise.
- Always keep correctness checks enabled (default program behaviour).

## 6) Interactive debugging (optional)

If a job fails and you want to reproduce interactively:

```bash
kgpu start -g 1 -w
kgpu shell -a -g 1

# inside the container:
export OMP_TARGET_OFFLOAD=MANDATORY
nvidia-smi
./build-dev/awave -shape 64,64,64 -nsteps 20 -out_period 10 /tmp/awave-debug

exit
kgpu stop
```

## 7) After each tested change (local git only)

1. Update `Completed.md` (what changed + why + test results).
2. Update `report_content.md` (numbers/tables/figures you will use in the final report).
3. Commit locally:

```bash
git add .
git commit -m "Test: <what> (shape=..., A100, diff=0)"
```

No `git push` (per current workflow).

## 8) Common failure modes

- OpenMP silently runs on CPU:
  - Set `OMP_TARGET_OFFLOAD=MANDATORY` in your job env (already done in `run-*.yml` here).
- CUDA/OpenMP falls back to CPU:
  - You are not on a GPU node, or the GPU is not visible inside the container.
- KGPU templates using `$(...)` inside `bash -c`:
  - `$(...)` is *command substitution* in bash (not an environment variable).
  - Use a literal path like `build-dev/awave`, or a proper shell variable like `$KGPU_JOB_NAME`.
