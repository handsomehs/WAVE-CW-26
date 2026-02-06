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
