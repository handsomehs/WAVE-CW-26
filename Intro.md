# 项目简介：ASPP Wave (3D Wave Equation)

## 目标
- 将 3D 波动方程的 CPU 参考实现扩展到 GPU（CUDA 与 OpenMP offload）。
- 在保证正确性的前提下，最大化单卡 A100 (NVIDIA-A100-SXM4-40GB) 的性能表现。
- 提交一份 3 页以内的报告，说明性能上限、实现选择与 CUDA/OMP 的对比结论。

## 算法（核心数值方法）
- **模型**：3D 波动方程，显式二阶时空离散。
- **空间离散**：三维 7 点 Laplacian（中心 + 6 邻点）。
- **时间推进**：
  - `u_next = 2*u_now - u_prev + (dt^2/dx^2)*cs2*laplacian`
  - 含边界阻尼项（只在 x/y 边界层有效）。

## 数据结构与布局
- **uField**：维护 `prev/now/next` 三个 3D 缓冲，带 ghost 层（每维 +2）。
- **cs2**：声速平方场（仅用于计算）。
- **damp**：边界阻尼场（仅在边界层非零）。
- **输出**：每 `out_period` 步写入 HDF5（VTK 兼容），用于可视化与验证。

## 代码架构（关键模块）
- `wave_cpu.cpp`：CPU 参考实现（必须保持不变）。
- `wave_cuda.cu`：CUDA 实现（你需要修改的文件之一）。
- `wave_omp.cpp`：OpenMP offload 实现（你需要修改的文件之一）。
- `main.cpp`：统一驱动程序，**固定顺序执行 CPU → CUDA → OpenMP**。
- `h5io.*`：HDF5 输出管理。

## 执行流程（高层）
1. 解析参数（shape/dx/dt/nsteps/out_period）。
2. CPU 参考初始化与运行。
3. CUDA 与 OpenMP 版本以 CPU 状态为起点运行。
4. 每个版本输出性能统计与 HDF5。
5. CUDA/OMP 结果与 CPU 对比（容差 1e-8）。

## 关键约束
- **只能修改** `wave_cuda.cu` 与 `wave_omp.cpp`。
- 禁止留下 AI 痕迹（注释需专业简洁）。
- 正确性优先：GPU 输出必须与 CPU 对齐。
