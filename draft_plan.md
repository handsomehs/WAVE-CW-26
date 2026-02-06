一、目标与量化指标
- 主要目标
  - 正确性：GPU 与 CPU 参考在既定容差内一致；边界与输出时刻严格一致。
  - 性能：以 site-updates/s（SU/s）和有效带宽（字节/点 × SU/s）逼近 A100 HBM 带宽上限。
  - 工程质量：数据常驻、资源生命周期清晰、模块化、可移植、易分析。
- 量化指标
  - SU/s；有效带宽/峰值带宽比例。
  - Nsight Compute：DRAM 吞吐、L2 重放率、分支发散、寄存器占用/溢出、SM 占用。
  - Nsight Systems：核/拷贝/写盘时间线重叠情况；API 调用密度（Graphs 对比流）。
  - 启动开销占比（特别是小/中规模）。

二、算法与数据流（带宽上限推导的基础）
- 计算模板：三维 7 点 stencil，显式二阶时空。
- 朴素访问（不做片上复用时，双精度）：
  - 读 u.now 6 或 7 次、读 u.prev 1 次、读 cs2 1 次、读 damp 1 次（内域可省）、写 u.next 1 次，约 80–88 B/点/步。
- 可利用的结构性
  - cs2 仅随 k 变 → 压缩为一维 cs2_k[k] 或用公式重算。
  - damp 仅随 (i,j) 变 → 压缩为二维 damp_xy[i,j]；内域核不读 damp。
  - 目标把每点字节压到约 32–40 B/点（共享内存 x–y 平铺 + z 向寄存器滑动 + 系数降维/降精度）。

三、内存与数据架构（“零拷贝策略”的正确解读）
- 数据常驻：模拟开始一次性 H2D；时间推进期间不回传；仅在快照时 D2H；结束统一释放。
- 指针轮换：每步仅交换设备端 prev/now/next 指针，不做设备内 memcpy；与输出缓冲的物理身份一致。
- 术语澄清：这里的“零拷贝”是“常驻 HBM、最少 H2D/D2H”，不是映射主机内存给 GPU 直访。

四、CUDA 实现蓝图（wave_cuda.cu）
- 线程映射与基本优化
  - threadIdx.x 对应内存最快维（stride-1）；块形 x 维为 32 的倍数（如 32×4–8×1/2）；使用 grid-stride loop 兼容任意规模并提升 ILP。
  - 核参数指针 const __restrict__；小标量 Params 放常量内存；cs2_k/damp_xy 放全局内存只读路径。
- 分核与分域
  - 内域核：不读 damp、无分支。
  - 边界核：读 damp，应用阻尼；可预计算 inv=1/(1+d·dt)、num=1−d·dt，内核中仅乘加。
- 片上复用与流水
  - 共享内存平铺 x–y 平面（含 halo）+ z 向寄存器滑动窗口 r_prev/r_curr/r_next；每推进一层 z 仅新增 1 次全局读。
  - cooperative_groups 的 memcpy_async(tb, sh, gptr, shape) + wait(tb) 管理 global→shared 装载；在 SM 8.x 上映射到 cp.async；成熟后做 tile 双缓冲实现装载与计算重叠。
  - 必要时在 warp 内用 shuffle 做小范围邻点交换或规约，减少共享内存/同步开销。
- 并发与异步流水线（Streams/Events/Host 回调）
  - 创建非阻塞流：1 条计算流 + 最多 2 条拷贝流（受 async engine 个数限制，A100 常为 2）。
  - 快照周期：在拷贝流上对输出缓冲分块 D2H（pinned 主机内存）；事件表达“计算完成→拷贝开始→写盘开始”的依赖。
  - cudaLaunchHostFunc 在 D2H 完成后入队轻量主机任务（写盘/压队列），确保回调中不调用 CUDA API。
  - 所有短生命周期设备内存一律用 cudaMallocAsync/cudaFreeAsync 与所在流绑定，避免全设备同步。
- CUDA Graphs（首选的启动开销优化）
  - 图节点：内域核 →（依赖）边界核 →（条件节点）D2H 拷贝 →（可选）HostFunc 写盘。支持每步更新指针/尺寸参数；小规模步进收益显著。
  - 捕获式或显式构建均可；注意图捕获期间不能做 stream 同步，跨流依赖用事件。
- 协作式网格与持久化内核（备选）
  - 若极端追求小规模延迟，可评估 cooperative launch + grid.sync 的持久化内核版本；代价是所有 blocks 常驻（每 SM 仅 1 block），吞吐可能下降。与 Graphs 二选一保留。
- 原子与诊断核
  - 原则：原子用于“稀疏/低争用”更新或汇总，不与非原子混用；能用更小作用域就不用更大（_block > _device > _system）。
  - 诊断/统计采用 warp/块内规约（cg::reduce/ballot/shuffle），仅在块尾做极少量全局原子；每 N 步单独小核计算 min/max/能量等，避免污染主核性能。
- 细节与坑
  - 默认流有全局栅栏语义，会破坏并行；一律用自建非阻塞流。
  - 异步 memcpy 需要主机端内存为 pinned 才真正异步。
  - memcpy_async 仅在 SM 8.x 的 global→shared 异步；否则退化为同步但语义正确。
  - Graph 捕获不能调用 cudaStreamSynchronize；用事件或图内依赖替代。

五、OpenMP 实现蓝图（wave_omp.cpp）
- 数据驻留与指针轮换
  - 类构造：target enter data map(alloc/to: …) 常驻；析构：exit data delete。
  - 计算阶段仅发 offload；输出时精确 update/from 需要的物理缓冲；与指针轮换保持一致。
- 并行与映射
  - target teams distribute parallel for collapse(3)；保证 stride-1 维由相邻线程处理以实现合并访问；调 num_teams、thread_limit、dist_schedule(static)。
  - 内/边界分开 offload；内域不读 damp。
- 异步与重叠
  - target nowait；输出周期用 nowait 的 update/from + task 依赖或 taskwait；主机端 I/O 与下一步 GPU 计算重叠。
  - 主机缓冲用 OpenMP allocator（pinned + alignment）或直接 CUDA pinned。
- 片上复用（进阶）
  - 使用 omp_pteam_mem_alloc 作为团队本地缓冲实现 x–y 平铺，z 向寄存器滑动；阶段性 barrier；视编译器实现决定投入产出。

六、数据与精度的工程化取舍
- 系数降维：cs2→一维 cs2_k[k]；damp→二维 damp_xy[i,j]；内域核不读 damp。
- 预计算阻尼：预存 inv 与 num，核内仅乘加。
- 计算替代存储（可选）：若 cs2(k) 可用低成本公式实时计算，省去读带宽。
- 混合精度（可选）：系数 float、状态 double；或全 float 的误差—性能曲线，确保满足容差。

七、工程与构建实践（来自“软件工程良好实践”课件）
- 接口与注解
  - 公共小函数尽量 __host__ __device__ inline；索引换算/物理系数轻量函数可 constexpr，启用 --expt-relaxed-constexpr。
  - 在核前注明需求清单：期望 blockDim/gridDim、共享内存字节、对齐/索引映射、架构要求（如 cp.async 需 sm_80+）。
- 代码与库
  - 需要 C++ 标准库在设备端的功能时，显式使用 libcudacxx（cuda::std::…），性能热点谨慎使用复杂抽象。
- 分离编译与链接
  - nvcc -dc/-c 做多翻译单元；最终由 nvcc 链接；或 CMake 打开 CUDA_SEPARABLE_COMPILATION。
- 多架构与 Fat binary
  - 至少生成 sm_80 并内嵌 compute_80（可另加 sm_90/compute_90）；CMake: -DCMAKE_CUDA_ARCHITECTURES="80;90"。
  - 调试/分析构建带 -lineinfo；编译诊断 -Xptxas -v 观察寄存器/溢出。
- 设备自适配
  - 启动时查询 SM 数、共享内存、寄存器、异步拷贝引擎数等；据此选择 BPG、块形、tile 尺寸与拷贝流数。
  - 用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 评估占用—寄存器—共享内存的折中。
- 资源与可观测性
  - 统一 CUDA_CHECK 宏；RAII 封装流/事件/内存；全链路 NVTX 标签覆盖 step/核/拷贝/写盘/图节点。

八、并发编排与 I/O 管线细化
- 双缓冲输出：两套设备/主机缓冲；达到 out_period 时把“要输出的那套物理缓冲”标记 D2H，GPU 继续下一步。
- 分块多流 D2H：按 Z 或连续内存块分段；两条拷贝流同时拉取，计算流继续核执行。
- 事件依赖：计算流 record → 拷贝流 wait → HostFunc 写盘入队；避免 host 阻塞同步。
- OpenMP 侧：nowait + 任务依赖表达；必要时专用 I/O 线程。

九、验证与性能评测设计
- Roofline 两档上限
  - 朴素：~80–88 B/点；优化：~32–40 B/点；对比实测与上限差距并分析（写分配、重放、非完美合并、占用等）。
- Nsight Systems
  - 对比单流串行/多流重叠/Graphs 三种时间线；观察 API 调用密度、空洞减少与拷贝-计算重叠。
- Nsight Compute
  - 关键指标：dram 吞吐、l2 重放率、gld/gst 合并度、分支发散、寄存器/共享内存、占用。
- 消融实验
  - 逐项开启：内/边界分核 → z 滑动 → x–y 平铺 → 系数降维/降精度 → 异步 I/O（多流）→ Graphs（或持久化）→ 分块 D2H 与 HostFunc；给出相对提升柱状图。
- 规模扫描
  - 小/中/大规模分段结论：小规模启动/延迟主导（Graphs 收益大）；大规模带宽主导（片上复用与系数压缩决定上限）。

十、正确性策略
- 基线对齐：32^3、64^3 小规模与 CPU 比对；再扩到大规模，重点检查内/边界交界与输出时刻。
- 指针轮换一致性：输出回传的始终是“当步要输出的物理缓冲”，避免被后续步覆盖。
- 防御性编程：所有 CUDA/OpenMP API 返回值检查；越界/非法尺寸提前报错。

十一、风险与规避清单
- 默认流的全局栅栏 → 一律使用非阻塞自建流。
- 原子与非原子混用 → 中间加同步或改为全原子；热点地址先做 warp/块内聚合。
- __syncthreads 前线程非对称退出 → 禁止早退路径跨 barrier。
- Graph 捕获中调用 stream 同步 → 用事件或图内依赖替代。
- 异步拷贝未用 pinned → 退化为同步；确保 host 缓冲锁页。
- 共享内存 bank 冲突/对齐 → 在 x 维按 32/64 倍数 padding；必要时小范围用 shuffle。
- 寄存器溢出 → 缩减临时变量、调小块形或 tile；关注 ptxas 报告。
- OpenMP 隐式映射反复传输 → 统一 enter/exit data 与 present 语义，避免隐式 from/to。

十二、优先级与里程碑（建议一周节奏，可按实际压缩/扩展）
- 里程碑1（第1–2天）：数据常驻 + 指针轮换 + 内/边界分核 + 正确的合并访问/并行形态；小规模正确性通过，拿到基线 SU/s。
- 里程碑2（第3天）：异步输出双缓冲（pinned 主机）+ 计算/拷贝分流 + 事件依赖；时间线出现稳定重叠。
- 里程碑3（第4天）：系数降维（cs2_k、damp_xy）+ 阻尼系数预计算；诊断核引入（warp/块内规约 + 少量原子）。
- 里程碑4（第5–6天）：z 滑动 + x–y 共享内存平铺；成熟后引入 memcpy_async 双缓冲；调块形/寄存器占用。
- 里程碑5（第7天）：CUDA Graphs（或对比持久化+grid.sync，择优一项保留）；小/中规模启动开销下降。
- 里程碑6（第8天）：规模扫描、消融实验、Roofline 对照、Nsight 截图与报告整理；工程清理与注释补全。

十三、交付清单（Definition of Done）
- 功能与正确性
  - 任意规模运行稳定，GPU/CPU 误差在容差内；输出无错位。
- 性能与并发
  - 大规模下有效带宽接近优化档 Roofline；时间线显示计算与 D2H 重叠；小规模下 Graphs 将启动开销显著压低。
- 工程与可维护性
  - 采用分离编译与 CMake；Fat binary（至少 sm_80 + compute_80）；NVTX 标签完备；统一错误检查与 RAII。
  - 设备自适配：根据属性决定 BPG/块形/tile/拷贝流数；cp.async 不可用时自动回退到同步 memcpy 路径，结果不变。
- 报告与证据
  - Roofline 推导、消融曲线、Nsight Systems/Compute 截图、参数与占用表、OpenMP vs CUDA 对比分析。
