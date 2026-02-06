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
- Nsight profiling：未执行。
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
  - 备注：当 `out_period` 很小（例如 2）时，`run()` 末的 D2H 回传在计时内，可能显著拉低 SU/s；因此性能测试建议使用较大的 chunk 长度（如 out_period=10）。

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
