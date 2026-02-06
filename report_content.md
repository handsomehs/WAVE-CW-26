# Report 可用内容素材（随实现逐步补充）

## Section 1: 理论性能上限估算（A100 Roofline）
- **核心公式**：
  - 每点 FLOPs（约 12–14，按实际 kernel 统计）
  - 每点字节流量（朴素 80–88 B；优化目标 32–40 B）
  - 带宽上限 SU/s ≈ HBM_BW / BytesPerSite
  - 计算上限 SU/s ≈ PeakFLOPs / FLOPsPerSite
  - 理论上限 = min(带宽上限, 计算上限)
- **需要补充的数据**：
  - A100 SXM4 40GB 官方带宽/峰值（引用 Nvidia 文档）
  - 实际 kernel 的 FLOPs/点与字节/点核算表

## Section 2: 实现选择与性能证据
- **设计选择（示例）**：
  - 数据常驻 + 指针轮换（避免每步 memcpy）。
  - 内/边界分核（减少分支）。
  - `cs2`/`damp` 降维或预计算（降带宽）。
  - OpenMP 与 CUDA 的并行映射策略。
- **性能证据（待补充）**：
  - SU/s 随规模变化曲线（32–1000）。
  - Nsight Systems 时间线截图（核/拷贝重叠）。
  - Nsight Compute 指标表（DRAM 吞吐、占用、分支发散）。
  - Roofline 证据：内域 kernel 的 Achieved DRAM Bandwidth 接近 A100 HBM 峰值（见下方“Profiling 数据点”）。

## 已收集的实测数据（可直接进报告）
- **测试说明**：
  - 程序固定执行 CPU → CUDA → OpenMP，并在每个 chunk 计时（I/O 不计入计时）。
  - OpenMP 运行设置 `OMP_TARGET_OFFLOAD=MANDATORY`，避免静默回退。
  - 测试用 YAML：`run-correct-mig.yml`、`run-perf-a100-128.yml`、`run-perf-a100.yml`、`run-perf-a100-256-20.yml`。
- **结果汇总（mean SU/s）**：
  - 64^3（A100 MIG 1g.5gb，nsteps=100,out_period=10）：CPU ≈ 3.25e8，CUDA ≈ 2.17e9，OpenMP ≈ 1.89e9。
  - 128^3（完整 A100，nsteps=20,out_period=10）：CPU ≈ 3.12e8，CUDA ≈ 4.12e9，OpenMP ≈ 3.67e9。
  - 256^3（完整 A100，nsteps=4,out_period=2）：CPU ≈ 2.59e8，CUDA ≈ 1.16e9，OpenMP ≈ 9.74e8。
- **分域优化后的新增数据（mean SU/s）**：
  - 64^3（A100 MIG 1g.5gb，nsteps=100,out_period=10）：CPU ≈ 3.21e8，CUDA ≈ 2.02e9，OpenMP ≈ 1.71e9。
  - 128^3（完整 A100，nsteps=20,out_period=10）：CPU ≈ 3.16e8，CUDA ≈ 4.74e9，OpenMP ≈ 4.07e9。
  - 256^3（完整 A100，nsteps=20,out_period=10）：CPU ≈ 2.90e8，CUDA ≈ 4.53e9，OpenMP ≈ 4.50e9。
- **系数压缩后的新增数据（mean SU/s，cs2->1D, damp->2D）**：
  - 64^3（A100 MIG 1g.5gb，nsteps=100,out_period=10）：CPU ≈ 3.19e8，CUDA ≈ 2.16e9，OpenMP ≈ 1.65e9。
  - 128^3（完整 A100，nsteps=20,out_period=10）：CPU ≈ 3.52e8，CUDA ≈ 4.23e9，OpenMP ≈ 3.62e9。
  - 256^3（完整 A100，nsteps=20,out_period=10）：CPU ≈ 2.94e8，CUDA ≈ 5.50e9，OpenMP ≈ 5.14e9。
- **将 D2H 回传移出计时区后的新增数据（mean SU/s，最终计时口径）**：
  - 64^3（A100 MIG 1g.5gb，nsteps=100,out_period=10）：CPU ≈ 3.21e8，CUDA ≈ 5.55e9，OpenMP ≈ 3.42e9。
  - 128^3（完整 A100，nsteps=20,out_period=10）：CPU ≈ 3.20e8，CUDA ≈ 3.08e10，OpenMP ≈ 2.24e10。
  - 256^3（完整 A100，nsteps=20,out_period=10）：CPU ≈ 2.87e8，CUDA ≈ 4.93e10，OpenMP ≈ 4.42e10。
- **规模扫描（完整 A100，最终计时口径，mean SU/s）**：
  - 说明：32^3/64^3/96^3 使用 `nsteps=200,out_period=100`；其余使用 `nsteps=20,out_period=10`（均为 2 chunks，输出写入 `/tmp` 并在 sweep job 中删除）。
  - 32^3：CPU ≈ 3.87e8，CUDA ≈ 2.49e9，OpenMP ≈ 8.83e8。
  - 64^3：CPU ≈ 3.49e8，CUDA ≈ 1.54e10，OpenMP ≈ 5.75e9。
  - 96^3：CPU ≈ 3.37e8，CUDA ≈ 3.42e10，OpenMP ≈ 1.51e10。
  - 128^3：CPU ≈ 3.08e8，CUDA ≈ 3.02e10，OpenMP ≈ 2.24e10。
  - 192^3：CPU ≈ 2.83e8，CUDA ≈ 4.33e10，OpenMP ≈ 3.65e10。
  - 256^3：CPU ≈ 3.03e8，CUDA ≈ 4.93e10，OpenMP ≈ 4.44e10。
  - 384^3：CPU ≈ 2.92e8，CUDA ≈ 5.32e10，OpenMP ≈ 4.97e10。
- **可写入报告的观察点**：
  - 小/中规模下 CUDA 与 OpenMP 均显著快于串行 CPU；CUDA 通常更高。
  - 规模增大后 SU/s 下滑，符合带宽受限 stencil 在缓存/带宽层级切换时的预期（需结合 Nsight 指标解释）。
  - 基线版本中 `run()` 计时包含每个 chunk 末尾的 D2H 回传；最终版本将回传移动到 `append_u_fields()`（计时外），使计时更接近纯计算（nsys NVTX 显示回传耗时远大于计算核）。

## Profiling 数据点（可直接进报告）
- **Nsight Systems（nsys）**：
  - profile 文件：`awave-nsys-20260206-095903.nsys-rep`（A100，shape=256^3，nsteps=4,out_period=4）
  - `nsys stats --report cuda_gpu_kern_sum` 的结论要点：
    - 计算时间几乎全部集中在 **内域 stencil kernel**（CUDA `step_kernel_interior` 与 OpenMP offload 对应内域 kernel）。
    - 边界层 kernel 的总耗时占比很小（每 step 的边界核仅 ~15–19 us 量级）。
  - `nsys stats --report nvtx_sum`（NVTX）要点：
    - `run`（4 steps 纯计算）≈ 1.72 ms
    - `copyback`（仅 CUDA 端 D2H）≈ 36.6 ms
- **Nsight Systems（nsys，小规模启动开销证据）**：
  - profile 文件：`awave-nsys-32-20260206-101912.nsys-rep`（A100，shape=32^3，nsteps=200,out_period=100）
  - `nvtx_sum`：
    - `run`（每 chunk 100 steps）≈ 1.56–1.71 ms
    - `copyback`（每 chunk）≈ 0.13 ms
  - `cuda_gpu_kern_sum`：
    - CUDA 内域/边界核均为 **~3 us/launch** 量级（200 次）
    - OpenMP 的 3 个 offload kernel 也在 **~3.6 us/launch** 量级（200 次）
  - 结论：小规模下 kernel 本身非常短，launch/运行时开销占比高；解释了 32^3/64^3 上 CUDA/OMP 的 SU/s 明显低于大规模带宽饱和区间。
- **Nsight Compute（ncu，聚焦 CUDA 内域 kernel）**：
  - profile 文件：`awave-ncu-20260206-095915.ncu-rep`
  - Roofline（`SpeedOfLight_RooflineChart`）核心数据：
    - Achieved **DRAM Bandwidth ≈ 1.37 TByte/s**
    - Workload achieved **~9% FP64 peak**（典型带宽受限行为）

## Section 3: CUDA vs OpenMP 对比
- **维度**：
  - 性能：大规模带宽饱和度、小规模启动开销。
  - 开发成本：代码复杂度、调优空间。
  - 可移植性：OpenMP 的可迁移性 vs CUDA 的生态与性能上限。
- **建议结论模板**：
  - 若以性能为优先 → CUDA。
  - 若以可移植与开发效率为优先 → OpenMP。
  - 提供量化证据支撑结论。

## 附：可直接引用/改写的技术描述模板
- “该 stencil 的算术强度低于 0.2 FLOP/Byte，因此性能上限由内存带宽决定。”
- “分离内域/边界核后，内域去除分支，提升了 warp 执行效率。”
- “通过数据常驻与指针轮换，避免了每步 H2D/D2H 开销，仅在输出点回传。”
- “OpenMP offload 在开发效率上更直接，但细粒度性能调优不如 CUDA 灵活。”
