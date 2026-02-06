# 计划（结合 draft_plan.md 与代码分析）

## 0. 项目理解与约束（必须牢记）
- **只能修改**：`src/wave_cuda.cu` 与 `src/wave_omp.cpp`。其他文件必须保持原样。
- **正确性检查**：程序会对 CUDA/OMP 的 `u.now()` 与 CPU 参考逐点比较（`approxEq`, eps=1e-8）。
- **运行流程**：`main.cpp` **固定依次运行 CPU → CUDA → OpenMP**，没有后端参数开关。GPU 实现必须在 `run()` 返回前，把当前场写回主机以供 HDF5 输出与对比。
- **数据结构**：
  - `uField` 维护 3 个环形缓冲（prev/now/next），每维 **有 2 层 ghost**。
  - `cs2` 与 `damp` 尺寸为 `nx*ny*nz`（无 ghost）。
- **性能瓶颈**：3D 7 点 stencil，双精度，算术强度极低（≈0.14 FLOP/Byte）→ **带宽受限**。

## 1. 目标与量化指标
- **正确性**：GPU 与 CPU 在容差内一致；输出时刻与 CPU 版本一致。
- **性能**：
  - 以 SU/s（site updates per second）和有效带宽逼近 A100 HBM 上限。
  - 小规模关注启动开销，大规模关注带宽饱和。
- **工程质量**：
  - 数据常驻、指针轮换、资源生命周期清晰；注释专业、简洁。
  - 禁止留下 AI 痕迹（README 要求）。

## 2. 算法与数据流（与 Roofline 推导相关）
- **Stencil**：三维 7 点 Laplacian + 阻尼分支。
- **朴素流量**（双精度）：
  - 读：`u.now`（6邻 + 中心）、`u.prev`、`cs2`、`damp`；写：`u.next`。
  - 约 80–88 B/点/步。
- **结构性优化空间**：
  - `cs2` 仅依赖 k：可压缩为一维 `cs2_k[k]`。
  - `damp` 只依赖 (i,j)：可压缩为二维 `damp_xy[i,j]`，内域不读。

## 3. 数据与内存架构（正确理解“零拷贝”）
- **数据常驻**：初始化时 H2D 一次；时间推进不回传；**仅输出时 D2H**。
- **指针轮换**：设备端 prev/now/next 指针交换，不进行 memcpy。
- **输出策略**：`run(len)` 完成后将当步 `u.now/u.prev` 写回主机。
- **异步输出前提**：若做计算/拷贝重叠，主机输出缓冲需 pinned。

## 4. CUDA 实现路线（wave_cuda.cu）
### 4.1 基线实现（必须完成）
- 分配设备内存：`u_prev/u_now/u_next`、`cs2`、`damp`。
- 简单 kernel（global memory 直读）：映射 `(i,j,k)` → `u` 索引（含 ghost +1）。
- 每步 kernel 后交换指针；`u.advance()` 更新时间。
- `run()` 末 `cudaMemcpy` 把 `u.now/u.prev` 传回主机。

### 4.2 分域与分核（提升性能/减少分支）
- **内域核**：`d==0`，不读 `damp`；无分支。
- **边界核**：读取 `damp` 并应用阻尼。

### 4.3 带宽优化（可选但高收益）
- `cs2` 压缩为 1D、`damp` 压缩为 2D（仅 GPU 侧压缩，不改 CPU 数据）。
- 内域核不访问 `damp`。
- 优先保证合并访问：`threadIdx.x` 对应 k 维（stride-1）。

### 4.4 进阶优化（按时间/收益选择）
- x–y 共享内存平铺 + z 寄存器滑动。
- 小规模启动开销优化：CUDA Graphs（首选）。
- 异步输出（多流 + pinned host）：与计算重叠。
- NVTX 标记 step/run/核/拷贝，便于 Nsight 可视化。
### 4.5 可选高级策略（时间允许再做）
- CUDA Graphs 之外，可评估持久化内核（cooperative launch + grid.sync）。二选一保留。
- 若采用 `cs2/damp` 压缩，确保数值一致性验证通过。

## 5. OpenMP Offload 实现路线（wave_omp.cpp）
### 5.1 基线实现
- `target enter data` 常驻 map；`target exit data` 回收。
- `target teams distribute parallel for collapse(3)` 实现 step。
- 输出周期 `target update from` 回传 `u.now/u.prev`。
- 运行期设置 `OMP_TARGET_OFFLOAD=MANDATORY` 防止静默回退。

### 5.2 优化路径
- 内/边界分核；内域不读 `damp`。
- `num_teams`、`thread_limit` 调参，保证 k 维合并访问。
- 输出期使用 `nowait` + 任务依赖重叠 I/O。

## 6. 正确性验证与测试流程
- **小规模回归**：32³、64³ → CPU 对比，无误差。
- **规模扫描**：128³、256³、512³、768³、1000³（注意显存）。
- **注意**：`main.cpp` 固定顺序执行 CPU/CUDA/OMP，无法仅跑单一后端。
- **提示**：若发现非立方体规模下对比异常，优先用立方体规模定位问题（当前 checker 使用 L 作为 k 上界）。

## 7. 构建与运行（环境流程）
- **构建**：`cmake -S src -B build-dev -DCMAKE_BUILD_TYPE=Release` → `cmake --build build-dev`。
- **运行**：`build-dev/awave -shape 32,32,32`（程序自动运行 CPU/CUDA/OMP）。
- **集群作业**：`run.yml` 默认 MIG；性能测试需改为完整 A100 节点。

## 8. Profiling 与性能评测
- **Nsight Systems**：检查 kernel/拷贝/写盘重叠与 API 密度。
- **Nsight Compute**：DRAM 吞吐、L2 重放率、占用、寄存器溢出。
- **Roofline**：
  - 朴素 ~80–88 B/点；优化目标 ~32–40 B/点。
  - 对比实测 SU/s 与理论带宽上限。

## 9. 报告准备与证据清单
- Section 1：基于 A100 规格与字节/点，推导理论上限 SU/s。
- Section 2：
  - 设计选择 → 性能证据（Nsight 图 + SU/s 曲线）。
  - 规模扫描与上限对比。
- Section 3：CUDA vs OpenMP 对比（性能、开发成本、可移植性）。

## 10. 风险与规避
- **显存临界**：1000³ 接近 40GB → 只在全量 A100 测试，必要时启用系数压缩。
- **默认流栅栏**：使用自建非阻塞流。
- **OpenMP 隐式映射**：必须显式 enter/exit data。
- **MIG 限制**：`run.yml` 默认 MIG 1g.5gb，性能测试需改用完整 A100 节点。

## 11. 里程碑（建议顺序）
1) **基线 CUDA/OMP 正确性**（通过小规模对比）
2) **性能基线**（记录 SU/s、带宽）
3) **分域优化**（内/边界分核）
4) **系数压缩**（1D/2D）
5) **异步输出/Graphs（可选）**
6) **规模扫描 + 报告整理**

## 12. 文档与提交纪律
- 每完成一个里程碑并测试后：更新 `Completed.md` / `report_content.md`。
- 按要求执行：`git add .` → `git commit -m "..."` → `git push`。
