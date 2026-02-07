# ASPP Coursework 1 Report (中文写作稿)

> 报告按 `README.md` 要求包含 3 个部分：**(1) 理论性能估算**、**(2) 实现选择与性能证据**、**(3) CUDA vs OpenMP 对比与推荐**。无需写引言/结论。
>
> 说明：本稿引用的数据来自本目录中的源码与测试/Profiling 产物（例如 `src/wave_cpu.cpp`、`src/main.cpp`、`src/wave_cuda.cu`、`src/wave_omp.cpp`、`report_content.md`、`run-sweep-a100-kernel-modes.yml`、`awave-nsys-*.nsys-rep`、`awave-ncu-*.ncu-rep`）。PDF要求为 10pt、<=3 页。

---

## 1) 单卡 A100 的理论可达性能估算（site updates/s）

### 1.1 算法与每点更新（site update）定义
本程序求解 3D 波动方程的显式二阶差分格式。对每个网格点 \((i,j,k)\) 的一步时间推进（即一次 *site update*），核心计算来自 `src/wave_cpu.cpp` 的 `step()`：

1. 7 点 stencil 拉普拉斯近似：
\[
\Delta u \approx u_{i-1,j,k}+u_{i+1,j,k}+u_{i,j-1,k}+u_{i,j+1,k}+u_{i,j,k-1}+u_{i,j,k+1}-6u_{i,j,k}
\]
2. 系数缩放（\(c^2\) 为空间变速声速平方，`cs2`）：
\[
\text{value} = \left(\frac{\Delta t^2}{\Delta x^2}\right)c^2 \Delta u
\]
3. 时间推进（无阻尼区域 \(d=0\)）：
\[
u^{n+1} = 2u^n - u^{n-1} + \text{value}
\]
4. x/y 边界层加入阻尼（\(d>0\)）：
\[
u^{n+1} = \frac{2u^n - (1-d\Delta t)u^{n-1} + \text{value}}{1 + d\Delta t}
\]
其中阻尼场 `damp(i,j,k)` 只在 x/y 的边界层非零，体区域严格为 0（见 `src/wave_cpu.cpp` 的阻尼初始化逻辑）。

### 1.2 Roofline：先判断算力还是带宽受限
以体区域（\(d=0\)）为主，其每点浮点操作数可按源码逐项统计为约 **12 FLOPs/site**（6 邻点求和 + 线性组合 + 两次乘法缩放 + 二阶时间推进）。边界层包含一次除法与更多乘加，但其体积分数很小（仅 x/y 的薄层），对总平均 FLOPs 影响有限。

若以 A100（`NVIDIA-A100-SXM4-40GB`）的公开规格：
- 峰值 FP64（非 Tensor）约 **9.7 TFLOP/s**；
- HBM2 带宽约 **1555 GB/s**；

则**计算上限**（假设全是 12 FLOPs/site）为：
\[
\text{SU/s}_{\text{compute}} \approx \frac{9.7\times 10^{12}}{12} \approx 8.1\times 10^{11}\ \text{site/s}
\]

要估计**带宽上限**，关键是每次 site update 的 DRAM 字节流量。朴素上需要读取 `u_now` 的 6 邻点 + 中心点、读取 `u_prev` 中心点、读取系数并写回 `u_next`，按双精度（8B）计为约 80–88B/site；但在 GPU 上邻点复用与 L2 缓存会显著降低 *实际* DRAM 流量。我们用 Nsight Compute（见 2.3）对内域 stencil kernel 的 roofline 结果做反推：在 256^3 的代表性配置下测得 **Achieved DRAM Bandwidth ≈ 1.37 TB/s**，结合程序输出的 SU/s（\(\sim 5\times 10^{10}\) 量级）可推得该 kernel 的**有效 DRAM 字节数约 26B/site**（数量级上接近“每点 1 次读 `u_now` + 1 次读 `u_prev` + 1 次写 `u_next`”，系数访问多在缓存中命中）。

因此带宽上限估算为：
\[
\text{SU/s}_{\text{BW}} \approx \frac{1.555\times 10^{12}}{26} \approx 6.0\times 10^{10}\ \text{site/s}
\]

综合得到单卡 A100 的理论上限：
\[
\text{SU/s}_{\max} \approx \min(\text{SU/s}_{\text{compute}},\text{SU/s}_{\text{BW}})\approx 6\times 10^{10}\ \text{site/s}
\]
即：该 stencil 在 A100 上应是**典型带宽受限**工作负载，合理的目标是在大规模问题上逼近 \(\sim 6\times 10^{10}\) site updates/s。

---

## 2) 实现选择、优化原因与性能证据（32–1000 尺度）

### 2.1 实验设置与计时口径（保证证据可复现）
- 程序固定执行顺序：CPU 参考 → CUDA → OpenMP offload，并用 `eps=1e-8` 将两种 GPU 结果与 CPU 参考比较（`src/main.cpp` 的 `checker`），本工作所有 sweep 中均得到 `Number of differences detected = 0`。
- 性能指标：`site updates/s` = \((nx\cdot ny\cdot nz\cdot \text{steps})/t\)，由程序对每个 chunk 的 `run()` 计时得到（`append_u_fields()` 在计时外执行，因此 I/O 不计入）。
- GPU 性能测试使用完整 A100（`NVIDIA-A100-SXM4-40GB`），批量扫频脚本为 `run-sweep-a100-kernel-modes.yml`；OpenMP 路径设置 `OMP_TARGET_OFFLOAD=MANDATORY` 防止静默回退到 CPU。

### 2.2 关键实现选择（CUDA 与 OpenMP 共同思路）
下面的选择都直接针对“带宽受限 + 小规模启动开销明显”的特征，并在 profiling / sweep 中验证。

1) 数据常驻设备端 + 指针轮换（避免每步 memcpy）
- CUDA：`cudaMalloc` 分配 `d_prev/d_now/d_next`，初始化一次 H2D；每步仅 kernel 写 `d_next`，然后交换指针。
- OpenMP：`omp_target_alloc` 常驻 `d_prev/d_now/d_next`，初始化一次 `omp_target_memcpy`；每步 offload 直接读写 device pointer，再轮换。
- 结果：将每步开销集中在 stencil kernel 上，避免把 PCIe/D2H/H2D 变成瓶颈。

2) 系数压缩：`cs2(i,j,k)`→`cs2_k[k]`，`damp(i,j,k)`→`damp_xy[i,j]`
- 依据：初始化阶段 `cs2` 仅随深度 k 变化（`src/wave_cpu.cpp` 里对每个 k 计算声速并填满 i/j），`damp` 仅随 (i,j) 变化（对 z 方向整列填充）。
- 做法：构造 1D `cs2_k`（nz）与 2D `damp_xy`（nx*ny）并一次性传到设备；kernel 中只读压缩数组。
- 作用：减少系数数组体积与 DRAM 访问压力；`cs2_k` 极小，`damp_xy` 在 (L,64,L) 等形状下也仅数 MB，容易驻留 L2，从而把带宽预算更多留给 3 个 `u` 场。

3) 将 D2H copyback 放到计时区外（与主程序“排除 I/O”一致）
- 主程序在每个 timed chunk 后调用 `append_u_fields()` 写输出；因此**理想的计时应只覆盖 `run()` 内的计算**。
- CUDA 与 OpenMP 均在 `append_u_fields()` 中做必要的 D2H，以保证写文件与最终比较使用的 host 数据正确，而 `run()` 只做计算与同步。
- Nsight Systems 证据（NVTX）表明：若 copyback 落在计时区内会严重“掩盖”真实计算性能（见 2.3）。

4) 预热（warm-up）消除首个 chunk 的一次性开销
- CUDA：初始化完成后执行一次极小的 `warmup_kernel<<<1,1>>>`，触发 module loading / runtime 初始化；
- OpenMP：初始化中执行一次极小的 `target` offload；
- 作用：减少第一段计时偏慢导致的 mean/std 偏差，提高报告中数据的可信度与可复现性。

5) 1/2/3 kernel（或 1/2/3 offload）策略扫频，并选择最稳健的默认策略
- 通过环境变量 `AWAVE_KERNEL_MODE=1/2/3` 强制选择：
  - mode 1：全域单 kernel/offload（每点分支判断是否阻尼）；
  - mode 2：两段（x 边界单独；其余合并）；
  - mode 3：三段（内域 + x 边界 + y 边界）。
- 在 A100 上对 32..1000 尺度做全量扫频（`run-sweep-a100-kernel-modes.yml`），结果显示：
  - CUDA：mode1 在 14/15 形状最优，mode3 仅在 384^3 领先约 2%；
  - OpenMP：mode1 在 15/15 形状最优；
  - 因此最终默认 **auto=mode1**，mode2/3 仅保留作对照实验。

其原因（与实测一致）是：阻尼层仅占很小比例，分支代价小；而额外 kernel/offload 启动带来的运行时开销在小/中规模很显著，且边界核的并行规模小、占用率低，综合上不如单核稳健。

### 2.3 Profiling 证据（为什么瓶颈在带宽/启动开销）

Nsight Systems（`nsys`）：
- 256^3 的代表性 profile（`awave-nsys-20260206-095903.nsys-rep`）NVTX 统计：
  - `run`（4 steps 纯计算）≈ **1.716 ms**
  - `copyback`（仅 D2H，同步）≈ **36.6 ms**
  说明：copyback 明显大于计算核，因此必须放在计时区外，才能用 SU/s 评价计算核本身。
- 32^3 小规模 profile（`awave-nsys-32-20260206-112413.nsys-rep`）kernel 汇总：
  - CUDA `step_kernel` 平均约 **3.77 us/step**
  - OpenMP 对应 offload kernel 平均约 **4.36 us/step**
  说明：单步计算只有微秒级，此时额外的 kernel/offload 启动次数会显著影响总时间，因此 mode1（单核）更优。

Nsight Compute（`ncu`）：
- `awave-ncu-20260206-095915.ncu-rep`（256^3 内域 kernel 的 roofline）核心结论：
  - Achieved DRAM Bandwidth ≈ **1.37 TB/s**
  - Workload 达到约 **9% FP64 峰值**
  说明：性能主要由 DRAM 带宽限制，而非计算单元不足；与第 1 部分的 Roofline 判断一致。

### 2.4 32–1000 尺度的性能结果（mode1，mean SU/s）
下表摘自 A100 sweep（`run-sweep-a100-kernel-modes.yml` 的 mode1 结果；不同尺寸用不同 nsteps/out_period 以获得稳定统计，但 SU/s 已按 “网格点数×步数/时间” 归一化，可横向比较）：

> 说明：从 512 开始使用形状 **(L, 64, L)**（而非 L^3）来覆盖到 L=1000，同时把总网格点数/输出体积控制在可测范围内；并且由于 `src/main.cpp` 的 checker 循环对 k 维上界使用了 `nx+2`（隐含假设 **nx==nz**），因此非立方测试保持 `nx==nz` 以避免误判。

| shape | CPU (SU/s) | CUDA (SU/s) | OpenMP (SU/s) | OpenMP/CUDA |
| --- | --- | --- | --- | --- |
| 32^3 | 3.45e8 | 6.54e9 | 2.31e9 | 0.35 |
| 64^3 | 3.37e8 | 3.04e10 | 1.42e10 | 0.47 |
| 96^3 | 3.36e8 | 5.19e10 | 2.85e10 | 0.55 |
| 128^3 | 3.03e8 | 4.64e10 | 3.52e10 | 0.76 |
| 160^3 | 2.84e8 | 4.80e10 | 4.06e10 | 0.85 |
| 192^3 | 2.87e8 | 4.94e10 | 4.34e10 | 0.88 |
| 224^3 | 2.83e8 | 5.13e10 | 4.62e10 | 0.90 |
| 256^3 | 2.90e8 | 5.21e10 | 4.76e10 | 0.91 |
| 384^3 | 2.68e8 | 5.26e10 | 4.95e10 | 0.94 |
| 512×64×512 | 3.00e8 | 5.10e10 | 4.66e10 | 0.91 |
| 640×64×640 | 2.74e8 | 5.20e10 | 4.77e10 | 0.92 |
| 768×64×768 | 2.74e8 | 5.27e10 | 4.70e10 | 0.89 |
| 896×64×896 | 2.72e8 | 5.30e10 | 4.89e10 | 0.92 |
| 1000×64×1000 | 2.72e8 | 5.32e10 | 4.91e10 | 0.92 |

主要观察与解释：
- **小规模（32–96）**：CUDA/OpenMP 明显受 kernel/offload 启动与运行时开销影响；CUDA 更轻量，因此领先更明显（OpenMP/CUDA≈0.35–0.55）。
- **中大规模（>=128）**：CUDA SU/s 稳定在 \(\sim 4.6\times 10^{10}\) 到 \(\sim 5.3\times 10^{10}\)，OpenMP 在 \(\sim 3.5\times 10^{10}\) 到 \(\sim 4.9\times 10^{10}\)，表现出**带宽饱和**特征；OpenMP 与 CUDA 的差距缩小到约 6–12%。
- **与第 1 部分理论上限对比**：大规模下 CUDA 约 \(5.3\times 10^{10}\) SU/s，接近带宽上限估算 \(6.0\times 10^{10}\) SU/s（约 85–90% 量级），与 Nsight Compute 的高带宽利用率一致。

---

## 3) CUDA vs OpenMP offload：性能、实现难度与推荐

### 3.1 性能对比（基于 32–1000 的 sweep）
- CUDA 在所有测试形状上都更快，优势在小规模最明显（32^3 时 OpenMP 约为 CUDA 的 35%）。
- 在更接近评测重点的大规模区间（>=128），OpenMP 通常可达到 CUDA 的 **88–94%** 左右（例如 256^3：0.91；1000×64×1000：0.92），说明两者最终都受同一个带宽上限约束，差距主要来自 offload 运行时与编译器生成代码的细节。

### 3.2 实现与调优成本
OpenMP target offload：
- 优点：与 CPU 代码结构接近，易于快速得到正确结果；同一份代码理论上更具可移植性（不同加速器/编译器实现）。
- 缺点：offload 运行时开销更显著；对线程块形状、内存层次（共享内存/只读缓存等）的可控性较弱；性能可解释性较依赖编译器。

CUDA：
- 优点：对并行映射（block/thread）、内存与同步更可控；Nsight 生态成熟，便于用 profiling 驱动优化；在小规模/复杂调优场景往往能更接近硬件上限。
- 缺点：样板代码更多（内存管理、kernel 配置、错误检查）；可移植性主要局限于 NVIDIA GPU。

### 3.3 推荐（若只能选一种）
若只能选一种编程模型以覆盖评测范围（32–1000）并尽可能获取更高且更稳定的性能，我会选择 **CUDA**：
- 它在全尺度上都给出更高 SU/s，且对“小规模启动开销”的控制更直接；
- 对于本题这种明确的带宽受限 stencil，CUDA 更便于用 Nsight 进行精确归因（带宽 vs 启动开销 vs 分支/访存模式），从而更稳健地逼近 roofline 上限。

同时，本实验也表明 OpenMP offload 在大规模时性能已非常接近 CUDA（通常仅差 6–12%），若项目强调可移植与开发效率，OpenMP 也是很有竞争力的选择。

---

## 参考文献（在 PDF 中按课程要求补全）
建议至少包含（并在正文中用 [1][2]… 引用）：
1. NVIDIA. *NVIDIA A100 Tensor Core GPU*（SXM4 40GB 规格：HBM 带宽、FP64 峰值等）。
2. NVIDIA. *CUDA C++ Programming Guide*（内存层次、并行执行模型等）。
3. OpenMP ARB. *OpenMP Application Programming Interface*（target offload 语义与实现）。
4. NVIDIA. *Nsight Systems / Nsight Compute Documentation*（nvtx、roofline 指标解释）。
