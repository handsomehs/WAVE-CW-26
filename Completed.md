# 已完成内容记录

> 说明：此文件记录已经完成的工作，包含原理、实现细节与效果。未实际完成的部分将明确标注“未实现/未测试”。

## 1. 项目需求与约束梳理（已完成）
- **原理**：准确识别评测约束是后续实现正确性的前提。
- **实现细节**：
  - 解析 `README.md` 与核心源码，确认仅可修改 `wave_cuda.cu` / `wave_omp.cpp`。
  - 确认执行流程为 CPU → CUDA → OpenMP，且无后端选择参数。
  - 理解 `uField` 环形缓冲与 ghost 层要求。
- **效果**：形成可执行的开发边界与正确性条件。

## 2. 计算核与性能模型初步分析（已完成）
- **原理**：3D 7 点 stencil 算术强度低，性能上限由带宽决定。
- **实现细节**：
  - 逐行阅读 CPU `step()`，确认访存模式与分支（阻尼分支）。
  - 估算每点字节流量约 80–88 B（朴素）。
- **效果**：明确优化方向应集中在带宽与访存合并。

## 3. 初版项目计划与文档体系（已完成）
- **原理**：将实现路线、测试与报告内容系统化，便于迭代更新。
- **实现细节**：
  - 生成 `plan.md`、`Intro.md`、`report_content.md`。
  - 搭建后续更新 `Completed.md` 的记录格式。
- **效果**：完成项目文档骨架，后续可在每个里程碑更新。

## 4. 未完成部分（明确列出）
- 报告撰写：未开始。

## 5. CUDA / OpenMP 基线实现（已完成并通过基础测试）
- **原理**：实现数据常驻、指针轮换的 GPU 版本，保证与 CPU 的数值一致性。
- **实现细节**：
  - CUDA：在 `wave_cuda.cu` 中增加设备内存分配与拷贝、kernel 计算、指针轮换；`run()` 末同步回传 `u.now/u.prev`。
  - OpenMP：在 `wave_omp.cpp` 中使用 `omp_target_alloc/omp_target_memcpy` 常驻数据，`target teams distribute parallel for` 执行 kernel，指针轮换后回传输出。
  - 两者均保留 CPU fallback（无设备时用 CPU step），以便在无 GPU 环境跑通。
- **效果**：在 A100（MIG 与完整卡）上通过与 CPU 参考的数值对比（`Number of differences detected = 0`），并得到初步性能数据（见第 6 部分）。

## 6. GPU 正确性与性能测试（已完成基础版本）
- **测试方法**：
  - 使用 `kgpu` 提交作业；OpenMP 路径强制 offload：`OMP_TARGET_OFFLOAD=MANDATORY`。
  - 为避免输出文件过大影响 PVC，输出路径指向容器 `/tmp`（YAML 中已设置）。
- **正确性结果**：
  - 64^3（A100 MIG 1g.5gb）：CUDA/OMP 均与 CPU 参考一致（0 differences）。
  - 128^3、256^3（完整 A100）：CUDA/OMP 均与 CPU 参考一致（0 differences）。
- **初步性能（SU/s，程序内计时，排除 I/O）**：
  - 64^3（MIG，nsteps=100,out_period=10）：CUDA mean ≈ 2.17e9，OpenMP mean ≈ 1.89e9。
  - 128^3（A100，nsteps=20,out_period=10）：CUDA mean ≈ 4.12e9，OpenMP mean ≈ 3.67e9。
  - 256^3（A100，nsteps=4,out_period=2）：CUDA mean ≈ 1.16e9，OpenMP mean ≈ 9.74e8。

## 7. 内域/边界分域优化（已完成并实测）
- **原理**：
  - 阻尼场 `damp(i,j,k)` 仅在 x/y 两个方向的边界层非零，内域严格为 0（见 `wave_cpu.cpp` 中阻尼初始化）。
  - 因此可把更新分为两类：
    - **内域**：`d=0`，不需要读取 `damp`，也不需要分支。
    - **边界层**：`d>0`，统一走阻尼公式。
  - 这样能减少内域的全局内存读（去掉 `damp`）并减少分支相关开销。
- **实现细节**：
  - CUDA（`wave_cuda.cu`）：
    - 新增 3 个 kernel：`step_kernel_interior`、`step_kernel_xbound`、`step_kernel_ybound`。
    - 采用“并集覆盖且不重叠”的分域方式：
      - 内域：`i=[nbl,nx-nbl), j=[nbl,ny-nbl)`。
      - x 边界层：`i in [0,nbl) ∪ [nx-nbl,nx)`（包含四角）。
      - y 边界层（排除 x 边界以免重复写）：`i=[nbl,nx-nbl), j in [0,nbl) ∪ [ny-nbl,ny)`。
  - OpenMP offload（`wave_omp.cpp`）：
    - 对应实现 3 个 `target teams distribute parallel for`，分别覆盖内域、x 边界层、y 边界层。
    - 由于 NVHPC 对 `target` 区域内包含多个 `teams` 构造有限制，因此采用多次 `target teams ...` 的方式保证可编译运行。
- **正确性与效果（实测）**：
  - 64^3（A100 MIG 1g.5gb，nsteps=100,out_period=10）：CUDA/OMP 仍为 0 differences。
  - 128^3（完整 A100，nsteps=20,out_period=10）：CUDA mean ≈ 4.74e9，OpenMP mean ≈ 4.07e9（较基线提升）。
  - 256^3（完整 A100，nsteps=20,out_period=10）：CUDA mean ≈ 4.53e9，OpenMP mean ≈ 4.50e9（稳定高带宽）。
  - 备注：早期版本中 D2H 回传位于 `run()` 内，`out_period` 很小（chunk 很短）时会显著拉低 SU/s；该问题已在第 10 部分通过将回传移至 `append_u_fields()`（计时外）解决。

## 8. 系数压缩：`cs2`(1D) + `damp`(2D)（已完成并实测）
- **原理**：
  - `cs2(i,j,k)` 仅依赖深度 k（由声速剖面决定），在 i/j 方向上重复；`damp(i,j,k)` 仅依赖 (i,j)，在 k 方向上重复。
  - 将两者在 GPU 侧压缩为更低维度数组（`cs2_k[k]`、`damp_xy[i,j]`）可：
    - 显著减少系数数组的设备端存储与 H2D 拷贝体积。
    - 让系数更容易驻留在缓存中，降低每点系数读取的有效带宽开销。
    - 对阻尼边界层：同一 (i,j) 下不同 k 线程读取同一 `damp_xy`，硬件可广播，减少冗余访存。
- **实现细节**：
  - CUDA（`wave_cuda.cu`）：
    - 初始化时在主机上构造 `std::vector<double> cs2_k(nz)`（取 `cs2(0,0,k)`）与 `damp_xy(nx*ny)`（取 `damp(i,j,0)`），拷贝到设备。
    - kernel 侧用 `cs2_k[k]` 与 `damp_xy[i*ny+j]` 替代原 3D 系数数组读取。
  - OpenMP offload（`wave_omp.cpp`）：
    - 同样构造 `cs2_k` / `damp_xy` 并用 `omp_target_memcpy` 传至设备常驻。
    - 更新循环与分域 kernel 改用压缩系数寻址。
- **正确性与效果（实测）**：
  - 64^3（A100 MIG 1g.5gb，nsteps=100,out_period=10）：CUDA/OMP 仍为 0 differences。
  - 128^3（完整 A100，nsteps=20,out_period=10）：CUDA mean ≈ 4.23e9，OpenMP mean ≈ 3.62e9（该规模下结果有一定波动，需更多重复实验确认趋势）。
  - 256^3（完整 A100，nsteps=20,out_period=10）：CUDA mean ≈ 5.50e9，OpenMP mean ≈ 5.14e9（相比未压缩约 4.5e9 有明显提升）。

## 9. Nsight Profiling（已完成：nsys + ncu）
- **目的**：为报告提供“性能为何如此”的证据：瓶颈位置、kernel 占比、带宽利用率（roofline）。
- **nsys（时间线/Kernel 占比）**：
  - YAML：`profile-nsys.yml`（A100，shape=256^3，nsteps=4,out_period=4；内存需 >=8Gi，否则会 OOMKilled）
  - 产物：`awave-nsys-20260206-093807.nsys-rep`（已在 `.gitignore` 忽略，不提交）
  - `nsys stats --report cuda_gpu_kern_sum` 关键结论：
    - 计算主要集中在 **内域 kernel**：`step_kernel_interior`（CUDA）与对应的 OpenMP offload kernel（`wave_omp.cpp` 内域 target 区域）。
    - 边界层 kernel（x/y 边界）单次仅 ~15–19 us 量级，远小于内域（~289–312 us）。
    - OpenMP 内域 kernel 略慢于 CUDA 内域 kernel（同规模、同步数下约 8% 左右），但同属带宽受限型 stencil。
- **ncu（Roofline/带宽利用）**：
  - YAML：`profile-ncu.yml`（为减少干扰，设置 `OMP_TARGET_OFFLOAD=DISABLED`，只剖析 CUDA 内域 kernel）
  - 产物：`awave-ncu-20260206-093818.ncu-rep`（已在 `.gitignore` 忽略，不提交）
  - `ncu --import ... --page details --print-details all`（Roofline section）关键数据：
    - `step_kernel_interior` Achieved **DRAM Bandwidth ≈ 1.37 TByte/s**（接近 A100 HBM 峰值，证明整体为带宽瓶颈）
    - 工作负载达到 **约 9% FP64 峰值**（进一步说明不是算力瓶颈）

## 10. 将 D2H 回传移出计时区（已完成并实测 + profiling）
- **原理**：
  - `main.cpp` 的 benchmark 计时仅包围 `run(len)`，随后调用 `append_u_fields()` 写输出（I/O 被刻意排除在计时之外）。
  - GPU 版本必须在输出/对比前把数据回传到主机，但这一步属于“输出管线”，不应污染计算核的计时。
- **实现细节**：
  - CUDA（`wave_cuda.cu`）：
    - `run()` 仅负责推进 n 步并在末尾 `cudaStreamSynchronize`，保证计时覆盖 kernel 执行。
    - `append_u_fields()` 中执行 D2H（`u.now/u.prev`）并同步，然后写 HDF5。
    - 添加 NVTX range `copyback`，便于 nsys 量化回传开销。
  - OpenMP（`wave_omp.cpp`）：
    - `run()` 只做 offload 计算与指针轮换。
    - `append_u_fields()` 中用 `omp_target_memcpy` 回传 `u.now/u.prev` 再写 HDF5。
- **正确性**：CUDA/OMP 在 MIG 与完整 A100 上均保持 `0 differences`。
- **性能效果（mean SU/s，程序内计时，I/O 不计入计时；D2H 现在也不计入计时）**：
  - 64^3（A100 MIG 1g.5gb，nsteps=100,out_period=10）：CUDA mean ≈ 5.49e9，OpenMP mean ≈ 3.27e9。
  - 128^3（完整 A100，nsteps=20,out_period=10）：CUDA mean ≈ 3.05e10，OpenMP mean ≈ 1.57e10。
  - 256^3（完整 A100，nsteps=20,out_period=10）：CUDA mean ≈ 4.93e10，OpenMP mean ≈ 3.67e10。
- **profiling 证据（nsys + NVTX）**：
  - `awave-nsys-20260206-095213.nsys-rep`：
    - NVTX：`run` ≈ 1.79 ms（4 steps 纯计算），`copyback` ≈ 43.8 ms（仅 CUDA 端回传）→ 回传远大于计算核。
    - `cuda_gpu_mem_time_sum`：D2H 总计约 81 ms（4 次 * ~137 MB，每次 ~20 ms，含 CUDA 与 OpenMP 回传）。

## 11. OpenMP Offload 预热以消除首个 chunk 初始化开销（已完成并实测 + profiling）
- **原理**：
  - OpenMP target offload 的首次进入通常会触发一次性的运行时初始化/设备上下文建立/JIT 等开销。
  - 本项目计时以 chunk 为单位（`run(len)`），若首次 offload 的一次性开销落在第一个 chunk 内，会导致 OpenMP 的第一个 chunk 显著偏慢、方差增大，从而影响“取 best run”的公平性与可解释性。
- **实现细节**（`src/wave_omp.cpp`）：
  - 在 `OmpImplementationData` 构造函数中，完成设备内存分配与 H2D 拷贝后，执行一次“极小” offload：
    - `#pragma omp target teams distribute parallel for device(device) is_device_ptr(u_now)`
    - 循环仅 1 次，对 `u_now[0]` 做自赋值（`u_now[idx] = u_now[idx];`）。
  - 目的仅是触发一次性初始化与代码路径，不改变数值结果；后续进入 `run()` 的首个 chunk 更接近稳态表现。
- **正确性**：MIG 与完整 A100 上 CUDA/OMP 仍保持 `Number of differences detected = 0`。
- **性能效果（mean SU/s，程序内计时，I/O 与 D2H 均不计入计时）**：
  - 64^3（A100 MIG 1g.5gb，nsteps=100,out_period=10）：CUDA mean ≈ 5.55e9，OpenMP mean ≈ 3.42e9（OpenMP 首 chunk 波动明显减小）。
  - 128^3（完整 A100，nsteps=20,out_period=10）：CUDA mean ≈ 3.08e10，OpenMP mean ≈ 2.24e10（较预热前 ≈1.57e10 显著提高且更稳定）。
  - 256^3（完整 A100，nsteps=20,out_period=10）：CUDA mean ≈ 4.93e10，OpenMP mean ≈ 4.42e10（较预热前 ≈3.67e10 提升且更稳定）。
- **profiling 证据（更新版 nsys + ncu）**：
  - `awave-nsys-20260206-095903.nsys-rep`：
    - `nvtx_sum`：`run` ≈ 1.72 ms（4 steps 纯计算），`copyback` ≈ 36.6 ms（仅 CUDA 端 D2H），`initialise` ≈ 264 ms（包含初始化与系数拷贝等）。
    - `cuda_gpu_kern_sum`：CUDA 内域 `step_kernel_interior` ≈ 288 us/step；OpenMP 内域 kernel ≈ 312 us/step；边界核单次约 15–19 us。
  - `awave-ncu-20260206-095915.ncu-rep`（Roofline，CUDA 内域 kernel）：
    - Achieved **DRAM Bandwidth ≈ 1.37 TByte/s**，Workload achieved **~9% FP64 peak** → 仍为典型带宽受限 stencil。

## 12. 规模扫描（32--384）与小规模启动开销 Profiling（已完成并记录到报告素材）
- **原理**：
  - 报告需要覆盖 32--1000 的规模范围；同时需要解释“为什么小规模性能低、大规模接近带宽上限”。
  - 对小规模（32^3）来说，每 step 的 kernel 极短，launch 与 offload runtime 开销会成为主要瓶颈，必须用 nsys 证据支撑结论。
- **实现细节（测试/流程）**：
  - 新增 sweep job：`run-sweep-a100.yml`（完整 A100 上按多个 shape 顺序运行，并在每次运行后删除 `/tmp` 输出，避免占满空间）。
  - 新增小规模 profiling job：`profile-nsys-32.yml`（`nsys` 采集 32^3，便于定位 launch/运行时开销占比）。
- **正确性**：sweep 覆盖的所有规模 CUDA/OMP 均保持 `0 differences`。
- **规模扫描结果（mean SU/s，完整 A100，最终计时口径）**：
  - 32^3（nsteps=200,out_period=100）：CUDA ≈ 2.49e9，OpenMP ≈ 8.83e8。
  - 64^3（nsteps=200,out_period=100）：CUDA ≈ 1.54e10，OpenMP ≈ 5.75e9。
  - 96^3（nsteps=200,out_period=100）：CUDA ≈ 3.42e10，OpenMP ≈ 1.51e10。
  - 128^3（nsteps=20,out_period=10）：CUDA ≈ 3.02e10，OpenMP ≈ 2.24e10。
  - 192^3（nsteps=20,out_period=10）：CUDA ≈ 4.33e10，OpenMP ≈ 3.65e10。
  - 256^3（nsteps=20,out_period=10）：CUDA ≈ 4.93e10，OpenMP ≈ 4.44e10。
  - 384^3（nsteps=20,out_period=10）：CUDA ≈ 5.32e10，OpenMP ≈ 4.97e10。
- **profiling 证据（小规模 32^3）**：
  - `awave-nsys-32-20260206-101912.nsys-rep`：
    - `nvtx_sum`：CUDA `run` 每 chunk（100 steps）≈ 1.56–1.71 ms，`copyback` 每 chunk ≈ 0.13 ms。
    - `cuda_gpu_kern_sum`：CUDA 三个 stencil kernel 均约 **~3 us/launch**；OpenMP 的三个 offload kernel 约 **~3.6 us/launch**（各 200 次）。
  - 结论：小规模主要受 launch/offload 运行时开销影响；大规模则能更好地逼近带宽上限（与 256^3 的 ncu roofline 一致）。

## 13. 补齐到 1000 维度的测试 + 大规模 Profiling（已完成并记录到报告素材）
- **原理**：
  - 报告要求覆盖 32--1000 的规模范围。
  - 同时需要规避 `main.cpp` 中 checker 对非立方尺寸的 bug：`k` 循环上界误用 `L=nx+2`（应为 `N=nz+2`）。因此测试非立方尺寸时需保证 `nx == nz`。
- **实现细节（测试/流程）**：
  - 新增大尺度测试 job：`run-large-a100.yml`（形状为 `512x64x512`、`768x64x768`、`1000x64x1000`，均为 `nsteps=20,out_period=10`，每次运行后删除 `/tmp` 输出）。
  - 新增大尺度 profiling job：`profile-nsys-1000x64x1000.yml`（`nsys` 采集 `1000x64x1000`，对比 run 与 copyback 的量级关系）。
- **正确性**：上述规模 CUDA/OMP 均保持 `0 differences`。
- **性能效果（mean SU/s，完整 A100，最终计时口径）**：
  - 512x64x512：CUDA ≈ 4.78e10，OpenMP ≈ 4.27e10。
  - 768x64x768：CUDA ≈ 5.10e10，OpenMP ≈ 4.69e10。
  - 1000x64x1000：CUDA ≈ 4.22e10（chunk 间波动大；best≈5.29e10），OpenMP ≈ 4.83e10（更稳定）。
- **profiling 证据（大规模回传 vs 计算）**：
  - `awave-nsys-1000x64x1000-20260206-102717.nsys-rep`：
    - `nvtx_sum`：`run` 每 chunk（2 steps）≈ 2.58–3.13 ms；`copyback` 每 chunk ≈ 133–149 ms；初始化 ≈ 352 ms。
    - `cuda_gpu_kern_sum`：CUDA 内域 `step_kernel_interior` ≈ 1.01 ms/step；OpenMP 内域 kernel ≈ 1.09 ms/step；边界核开销相对更小但仍可见。
  - 结论：当场数据很大时，D2H 回传（为输出/对比所需）会主导端到端 walltime；而基准计时刻意排除 I/O，所以应将 copyback 置于 `append_u_fields()`（计时外）并用 NVTX 量化其成本。
