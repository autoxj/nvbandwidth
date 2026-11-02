

# nvbandwidth 参数与测试用例说明

> 参数定义位于 `nvbandwidth.cpp`（argparse 注册部分），默认值常量位于 `test_settings.h`。
> 测试用例注册位于 `nvbandwidth.cpp` 的 `createNodeTestcases()`，key 生成逻辑位于 `testcases.cpp` 与 `multinode_testcases.cpp`。

---

# 一、命令行参数说明

## 1. 公开参数（--help 可见）

| 参数 | 默认值 | 有效范围 | 作用 |
|---|---|---|---|
| `--version` | 关闭 | 标志 | 打印 nvbandwidth 版本号后退出 |
| `-l` / `--list` | 关闭 | 标志 | 列出所有可用 testcase 后退出 |
| `-b` / `--bufferSize SIZE` | 512（MiB） | uint64，无上限 | memcpy 缓冲区大小。小于 512 时仅输出警告（建议使用默认值以达到峰值带宽）。延迟测试固定使用 2 MiB，不受此参数影响 |
| `-t` / `--testcase TEST...` | 空（即跑全部） | 测试名或索引，可追加多次 | 按名称/索引指定要跑的测试；与 `-p` 互斥 |
| `-p` / `--testcasePrefixes PREFIX...` | 空 | 前缀字符串，可追加多次 | 按前缀批量选择测试（如 `-p device_to_device`）；与 `-t` 互斥，无匹配时报错退出 |
| `-v` / `--verbose` | 关闭 | 标志 | 输出详细日志信息 |
| `-s` / `--skipVerification` | 关闭 | 标志 | 跳过拷贝后的数据正确性校验 |
| `-d` / `--disableAffinity` | 关闭 | 标志 | 禁用自动 CPU 亲和性绑定 |
| `-i` / `--testSamples N` | 3 | uint32 | 基准测试重复采样次数（结果取中位数或均值）。≥3 时才输出变异系数（CV）稳定性指标 |
| `-P` / `--targetNumPairs N` | -1 | int64，≥ -1 | 多节点 device-to-device 测试的目标 GPU 对数。`-1`=所有对，`0`=不对，正数上限为 `worldSize×(worldSize-1)`，超出报错退出 |
| `-m` / `--useMean` | 关闭 | 标志 | 用均值代替中位数汇总结果 |
| `-H` / `--useHugePages` | 关闭 | 标志 | 主机内存使用大页。Windows 上忽略；Linux 需开启 THP，否则警告并忽略 |
| `-F` / `--format FORMAT` | text | text / json / perf | 输出格式 |
| `--pair UUID UUID` | 无 | 两个不同的 GPU UUID（nargs=2） | 只在指定的两块 GPU 之间跑测试（可跨节点）。单节点要求两块卡都在本机；多节点要求恰好 2 个 rank。仅对 device-to-device memcpy/latency 类测试生效 |

## 2. 隐藏参数（help 中不显示）

| 参数 | 默认值 | 有效范围 | 作用 |
|---|---|---|---|
| `-j` / `--json` | 关闭 | 标志 | 旧版 JSON 输出开关（向后兼容），等价 `--format json`；不能与 `--format` 同时使用 |
| `--loopCount N` | 16 | uint64 | 每个采样内执行的 memcpy 迭代次数 |
| `--disablePerformanceCache` | 关闭（缓存默认开启） | 标志 | 禁用双向测试结果缓存 |
| `--perfFormatter` | 关闭 | 标志 | 输出中附加 perf formatter 前缀 |
| `--disableNvlinkSMScheduling` | 关闭 | 标志 | 禁用 SM 测试的 NVLink 利用率调度（Blackwell 及之后架构） |

## 3. 默认值常量（test_settings.h）

```cpp
constexpr std::uint64_t kDefaultLoopCount = 16;
constexpr std::uint64_t kDefaultBufferSizeMiB = 512;  // Buffer size in MiB
constexpr std::uint32_t kDefaultAverageLoopCount = 3;
constexpr std::int64_t kDefaultTargetPairs = -1;      // -1 means all pairs
```

## 4. 补充说明

- 布尔型标志默认均为关闭。
- 不指定 `-t`/`-p` 时默认运行全部 44 个单节点测试（多节点模式下还会追加 19 个多节点测试）。
- 命令行示例：

```
nvbandwidth                          # 运行全部测试
nvbandwidth -l                       # 列出可用测试
nvbandwidth -t host_to_device_memcpy_ce device_to_host_memcpy_ce
nvbandwidth -p device_to_device      # 运行所有 device_to_device* 测试
nvbandwidth --format json            # JSON 格式输出
```

---

# 二、测试用例说明

## 1. 命名规则

`{方向}_{访问模式}_memcpy_{引擎}`，引擎后缀含义：

- **ce** = Copy Engine：统一使用 `cuMemcpyAsync`，驱动派发给 GPU 专用 DMA 拷贝硬件执行，不占用 SM
- **sm** = SM 拷贝 kernel：手写 CUDA kernel 在 SM 计算单元上跑拷贝代码。具体算子（见 `kernels.cu`）：
  - `stridingMemcpyKernel` —— 单向拷贝（普通 `_sm` memcpy 变体）
  - `splitWarpCopyKernel` —— 双向拷贝（奇偶 warp 交替方向，bidirectional `_sm` 变体）
  - `stridingReadOnlyKernel` / `stridingWriteOnlyKernel` —— 本地显存纯读 / 纯写
  - `ptrChasingKernelSM` —— 指针追逐测延迟
  - `multicastCopyKernel` —— 多播内存写入（multinode broadcast / fabric put）
- **tma** = Tensor Memory Accelerator：`tma_memcpy_kernel`（TMA 批量拷贝指令，仅 Blackwell+，仅设备间）；延迟测试用 `ptrChasingKernelTMA`

## 2. 单机测试（44 个）

### 2.1 主机 ↔ 设备（CE/SM 各 4 个，共 8 个）

| key 模式 | 含义 | 备注 | 接口/算子 |
|---|---|---|---|
| `host_to_device_memcpy_{ce,sm}` | 页锁定（pinned）主机内存 → GPU 带宽（按 NUMA 亲和分配） | 仅单机。逐卡**串行**测量（每轮仅一对 buffer、独占链路），结果为 1×N 矩阵。拷贝由 GPU 侧发起执行（host 不驱动 DMA） | ce：`cuMemcpyAsync`；sm：`stridingMemcpyKernel` |
| `device_to_host_memcpy_{ce,sm}` | GPU → 主机内存带宽 | 仅单机。同上，逐卡串行，1×N 矩阵 | ce：`cuMemcpyAsync`；sm：`stridingMemcpyKernel` |
| `host_to_device_bidirectional_memcpy_{ce,sm}` | 双向同时传输，只报告 H2D 方向带宽。CE 用双流（D2H 干扰流缓冲为 2 倍大小）；SM 用 split-warp kernel，奇偶 warp 交替拷贝方向 | 仅单机。测量期间同一链路上 H→D 与 D→H 流量**同时发生**（CE：GPU Copy Engine 同时执行 2 个 stream；SM：单个 kernel 内奇偶 warp 各拷一个方向）。**不是 2 个线程**——host 侧只有 1 个 CPU 线程负责提交，拷贝由 GPU 硬件执行。逐卡串行（每轮仅 1 host ↔ 1 GPU）。异步执行时序见[附录 A](#附录-ace-双流执行时序async-拷贝模型)；与 D2H 变体的执行差异见[附录 B](#附录-bh2d-bidir-与-d2h-bidir-的执行差异) | ce：`cuMemcpyAsync` ×2 流；sm：`splitWarpCopyKernel` |
| `device_to_host_bidirectional_memcpy_{ce,sm}` | 同上，只报告 D2H 方向带宽 | 仅单机。与上一条互为测量视角（测量方向与干扰方向互换），各自独立测量。执行时序见[附录 A](#附录-ace-双流执行时序async-拷贝模型)；两者执行差异见[附录 B](#附录-bh2d-bidir-与-d2h-bidir-的执行差异) | ce：`cuMemcpyAsync` ×2 流；sm：`splitWarpCopyKernel` |

> **"双向并发"的准确含义**：指同一块 GPU 与主机之间的链路上，两个方向的传输在同一时间窗内重叠执行，从而测出双向干扰下的单向带宽。并发发生在 GPU 硬件层面（DMA 引擎或 SM warp），host 侧不存在多线程。若要"host 同时对多块 GPU"，见 2.3 节的 `host_to_all_*` / `all_to_host_*` 系列。

### 2.2 设备 ↔ 设备（CE/SM/TMA 各 4 个，共 12 个）

| key 模式 | 含义 | 备注 | 接口/算子 |
|---|---|---|---|
| `device_to_device_memcpy_read_{ce,sm,tma}` | P2P 读（从远端 GPU 读到本地，读侧带宽） | 仅单机。默认**全网卡参与**，按行优先遍历全部有序对：GPU0→1、GPU0→2 … GPU0→7，再 GPU1→0、GPU1→2 …，共 N×(N-1) 次。严格**逐对串行**（每对独占链路），结果为 N×N 矩阵（行=src，列=peer），无对称去重——(0,1) 与 (1,0) 是两次独立测量。`--pair` 可收敛为单对。TMA 变体需 Blackwell+。串行/并行结构详见[附录 C](#附录-cd2d-nxn-矩阵的串行与并行结构) | ce：`cuMemcpyAsync`；sm：`stridingMemcpyKernel`；tma：`tma_memcpy_kernel` |
| `device_to_device_memcpy_write_{ce,sm,tma}` | P2P 写（本地写到远端 GPU，写侧带宽） | 仅单机。同上：全网卡参与、行优先遍历、逐对串行、N×N 矩阵、无对称去重。结构详见[附录 C](#附录-cd2d-nxn-矩阵的串行与并行结构) | ce：`cuMemcpyAsync`；sm：`stridingMemcpyKernel`；tma：`tma_memcpy_kernel` |
| `device_to_device_bidirectional_memcpy_read_{ce,sm,tma}` | 每对 GPU 同时双向拷贝，报告读侧分量 | 仅单机。遍历仍**逐对串行**，但每对内两路拷贝同时跑（双向并发即在此处）；对称 GPU 对 (src,dst)/(dst,src) 通过 cachedPairLoop 共用一次物理测量。详见[附录 C](#附录-cd2d-nxn-矩阵的串行与并行结构) | ce：`cuMemcpyAsync` ×2 流；sm：`splitWarpCopyKernel`；tma：`tma_memcpy_kernel` |
| `device_to_device_bidirectional_memcpy_write_{ce,sm,tma}` | 同上，报告写侧分量。同一测试内对称 GPU 对 (src,dst)/(dst,src) 共用一次测量 | 仅单机。与上一条互为测量视角，各自独立测量。详见[附录 C](#附录-cd2d-nxn-矩阵的串行与并行结构) | ce：`cuMemcpyAsync` ×2 流；sm：`splitWarpCopyKernel`；tma：`tma_memcpy_kernel` |

### 2.3 多 GPU 聚合模式（CE/SM 各 8 个，共 16 个）

| key 模式 | 含义 | 备注 | 接口/算子 |
|---|---|---|---|
| `all_to_host_memcpy_{ce,sm}` | 测某块 GPU → 主机的带宽，同时其余所有 GPU 也向主机并发拷贝作为干扰负载；报告被测 GPU 的带宽（每卡一个值） | 仅单机。多 GPU **并发**（干扰负载）；逐目标轮换测量；USE_FIRST_BW（只取被测那一路）。单线程实现与输出判读见[附录 D](#附录-d聚合干扰类测试的执行与输出all_to_host--host_to_all) | ce：`cuMemcpyAsync` 多流；sm：`stridingMemcpyKernel` 多流 |
| `all_to_host_bidirectional_memcpy_{ce,sm}` | 测某块 GPU 的 D2H，同时叠加该 GPU 的 H2D 及其余所有 GPU 的双向干扰流；只报告被测的 D2H 方向 | 仅单机。全网并发 + 本卡双向；USE_FIRST_BW。见[附录 D](#附录-d聚合干扰类测试的执行与输出all_to_host--host_to_all) | ce：`cuMemcpyAsync` 多流（含反向）；sm：`splitWarpCopyKernel` 多流 |
| `host_to_all_memcpy_{ce,sm}` | 测主机 → 某块 GPU 的带宽，同时主机向其余所有 GPU 并发拷贝作为干扰负载；报告被测 GPU 的带宽（每卡一个值） | 仅单机。多 GPU 并发干扰；逐目标轮换；USE_FIRST_BW。见[附录 D](#附录-d聚合干扰类测试的执行与输出all_to_host--host_to_all) | ce：`cuMemcpyAsync` 多流；sm：`stridingMemcpyKernel` 多流 |
| `host_to_all_bidirectional_memcpy_{ce,sm}` | 测主机 → 某块 GPU 的 H2D，同时叠加该 GPU 的 D2H 及其余所有 GPU 的双向干扰流；只报告被测的 H2D 方向 | 仅单机。全网并发 + 本卡双向；USE_FIRST_BW。见[附录 D](#附录-d聚合干扰类测试的执行与输出all_to_host--host_to_all) | ce：`cuMemcpyAsync` 多流（含反向）；sm：`splitWarpCopyKernel` 多流 |
| `all_to_one_write_{ce,sm}` | 所有 GPU → 单个 GPU（汇聚写入，报告每目标总入带宽，TOTAL_BW 聚合） | 仅单机。N2One 并发聚合；按目标逐一轮换，每目标一个 TOTAL_BW 值。**"all"组排除 one 自身**（实际 N-1 路）；P2P 不可达的卡跳过；某轮无任何可达 peer 则跳过该卡 | ce：`cuMemcpyAsync` 多流；sm：`stridingMemcpyKernel` 多流 |
| `all_to_one_read_{ce,sm}` | 所有 GPU 从单个 GPU 读（报告每源总出带宽，TOTAL_BW 聚合） | 仅单机。N2One（数据流 all→one 侧读取）；按源逐一轮换；TOTAL_BW。**"all"组排除 one 自身**（实际 N-1 路）；P2P 不可达的卡跳过 | ce：`cuMemcpyAsync` 多流；sm：`stridingMemcpyKernel` 多流 |
| `one_to_all_write_{ce,sm}` | 单 GPU → 所有 GPU（报告每源总出带宽，TOTAL_BW 聚合） | 仅单机。One2N 并发聚合；按源逐一轮换；TOTAL_BW。**目标组排除源自身**（实际 N-1 路）；P2P 不可达的卡跳过 | ce：`cuMemcpyAsync` 多流；sm：`stridingMemcpyKernel` 多流 |
| `one_to_all_read_{ce,sm}` | 单 GPU 从所有 GPU 读（数据流 all→one，报告每目标总入带宽，TOTAL_BW 聚合） | 仅单机。One2N（数据流 all→one）；按目标逐一轮换；TOTAL_BW。**目标组排除源自身**（实际 N-1 路）；P2P 不可达的卡跳过 | ce：`cuMemcpyAsync` 多流；sm：`stridingMemcpyKernel` 多流 |

### 2.4 本地带宽 / 延迟（8 个）

| key | 含义 | 备注 | 接口/算子 |
|---|---|---|---|
| `device_local_copy`（CE） | GPU 显存内部拷贝带宽（同一块卡） | 仅单机。逐卡串行测量，无跨卡流量 | `cuMemcpyAsync` |
| `device_local_copy_sm` | 同上，SM kernel 实现 | 仅单机。逐卡串行 | `stridingMemcpyKernel` |
| `device_local_copy_tma` | 同上，TMA 实现 | 仅单机。需 Blackwell+（TMA），否则跳过 | `tma_memcpy_kernel` |
| `device_local_read_sm` | SM kernel 读本地显存带宽（显存读速率） | 仅单机。逐卡串行，无跨卡流量 | `stridingReadOnlyKernel` |
| `device_local_write_sm` | SM kernel 写本地显存带宽（显存写速率） | 仅单机。逐卡串行 | `stridingWriteOnlyKernel` |
| `host_device_latency_sm` | 主机↔GPU 访问延迟（指针追逐 kernel，固定 2 MiB 缓冲） | 仅单机。缓冲固定 2 MiB，不受 `-b` 影响；逐卡串行 | `ptrChasingKernelSM` |
| `device_to_device_latency_sm` | GPU 对间指针解引用延迟（固定 2 MiB） | 仅单机。逐对串行，N×N 矩阵；固定 2 MiB | `ptrChasingKernelSM` |
| `device_to_device_latency_tma` | 同上，TMA 访问方式（需 TMA 支持） | 仅单机。需 Blackwell+；逐对串行 | `ptrChasingKernelTMA` |

> **device_local 系列的完整数据路径**：src 与 dst 是同一块 GPU HBM 上的两块独立显存分配。SM 不直接连 HBM，所有访存必经 L2 中转，因此一次 copy 的完整路径是：
>
> ```
> 读半程:  HBM(src 区域) → L2 → SM 寄存器      // load
> 写半程:  SM 寄存器 → L2 → HBM(dst 区域)      // store（经 L2 写回，异步落盘）
> ```
>
> 合成一条：`HBM(src) → L2 → SM → L2 → HBM(dst)`——全程在同一颗 GPU 芯片内完成，不经过 NVLink/PCIe 互连。"不出芯片"指数据不跨卡，而非不经过缓存。
>
> 默认 512 MiB buffer 远大于单卡 L2（几十 MB 量级），稳态下读写几乎全部穿透到 HBM，测得的吞吐 ≈ **HBM 有效带宽**。若把 `-b` 调小到 L2 容量以内，src/dst 常驻缓存，测的将变成 L2 带宽——这也是参数表中建议 512 MiB 的原因之一。
>
> copy / read / write 三个变体的差别：copy 同时产生读+写流量（混合吞吐）；read（`stridingReadOnlyKernel`）测纯读速率，write（`stridingWriteOnlyKernel`）测纯写速率，两者方向相反、数值通常不相等（写路径涉及 L2 回写、ECC 等额外开销）。

> **带宽测试（`_memcpy_`）与延迟测试（`_latency_`）的区别**：两者测同一链路上完全独立的两个指标，测法根本不同。
>
> | | 带宽测试 | 延迟测试 |
> |---|---|---|
> | 指标 | 吞吐能力（GB/s） | 单次访问往返时间（µs/ns） |
> | buffer | 512 MiB（`-b` 可调）大块流式 | 固定 2 MiB（不受 `-b` 影响） |
> | 测法 | 批量搬运，总字节 ÷ 总时间 | `ptrChasingKernelSM/TMA` 指针追逐 |
> | 访问模式 | 地址连续、预取友好、多请求并行在途 | 链式随机解引用，每步依赖上一步，无法预取 |
>
> 指针追逐原理：buffer 被组织成随机链表（每个位置存下一个位置的地址），kernel 逐级解引用 `p = *p`，总时间 ÷ 访问次数 = 平均延迟。因为下一步地址取决于上一步读到的值，GPU 的预取、多请求并行等延迟隐藏手段全部失效，每次访问都暴露完整往返时间。
>
> 带宽测试为何测不到延迟：512 MiB 连续大块传输时，DMA/CE 将传输深度流水化，几十上百个请求同时在途，单请求延迟被完全摊薄。反之延迟测试的链式单访问也无法体现吞吐。
>
> 用途分工：带宽对应批量数据搬运场景（权重加载、数据集上传）；延迟对应细粒度随机访问场景（GPU 零拷贝访问主机数据结构）。`*_memcpy_*` 后缀均为带宽版，`*_latency_*` 后缀均为延迟版，同一链路两者成对出现（host_device、device_to_device 各一对）。

## 3. 跨机（多节点）测试（19 个，仅 MPI 模式）

编译时定义 `MULTINODE` 且 `env->getSize() > 1` 才会注册，单机运行时这些测试不存在。

### 3.1 跨机设备间拷贝（CE/SM/TMA 各 4 个，共 12 个）

| key 模式 | 含义 | 备注 | 接口/算子 |
|---|---|---|---|
| `multinode_device_to_device_memcpy_read_{ce,sm,tma}` | 跨节点 GPU 对之间 P2P 读（走 NVLink/网络） | 需 MPI。逐对串行测量；rank 0 生成测试对并广播；数据通路为 fabric 显存映射（非 socket）；对数由 `-P` 控制；`--pair` 可指定单对；TOTAL_BW | ce：`cuMemcpyAsync`；sm：`stridingMemcpyKernel`；tma：`tma_memcpy_kernel` |
| `multinode_device_to_device_memcpy_write_{ce,sm,tma}` | 跨节点 P2P 写 | 需 MPI。同上，逐对串行；TOTAL_BW | ce：`cuMemcpyAsync`；sm：`stridingMemcpyKernel`；tma：`tma_memcpy_kernel` |
| `multinode_device_to_device_bidirectional_memcpy_read_{ce,sm,tma}` | 跨节点同时双向拷贝（读分量） | 需 MPI。单对内双向并发；逐对串行；TOTAL_BW | ce：`cuMemcpyAsync` ×2 流；sm：`splitWarpCopyKernel`；tma：`tma_memcpy_kernel` |
| `multinode_device_to_device_bidirectional_memcpy_write_{ce,sm,tma}` | 同上（写分量） | 需 MPI。同上；与读分量测试互为测量视角 | ce：`cuMemcpyAsync` ×2 流；sm：`splitWarpCopyKernel`；tma：`tma_memcpy_kernel` |

#### 补充：支持 M2N 参与方式（节点 A 任意 M 张 + 节点 B 任意 N 张），但为逐对串行执行

上表 4 组测试对参与卡的选取没有任何"必须 1 对 1"的限制：只要两个节点合计发起 M+N 个 MPI rank（每 rank 绑一张卡），就支持节点 A 任意 M 张卡与节点 B 任意 N 张卡参与测试，即 M2N 组合。参与的卡由 rank 分配控制（hostfile `slots` 或 wrapper 脚本注入 `CUDA_VISIBLE_DEVICES`），程序本身不感知"M2N"这个概念。

但要注意执行语义：测试不是"M 张卡和 N 张卡同时互灌"，而是把 M×N 个跨节点对**逐一串行**测量——每对轮流建 buffer、拷贝、测速，任何时刻网络上只有一张卡在发、一张卡在收。最终得到 M×N 个单对带宽数值（矩阵形式），适合逐链路分析和定位最慢路径。真正的并发聚合要看 3.2 的 all_to_one / all_from_one / bisect。

M2N 示例（nodeA 用 3 张卡、nodeB 用 2 张卡，共 5 rank）：

```bash
# hostfile:
# nodeA slots=3
# nodeB slots=2

# 方式一：每节点按序取前几张卡（nodeA GPU0-2，nodeB GPU0-1）
mpirun -np 5 -hostfile hosts ./nvbandwidth -t multinode_device_to_device_memcpy_read_ce

# 方式二：每节点任选指定卡（wrapper 控制可见卡集合，如 nodeA 用物理卡 1,3,5，nodeB 用物理卡 0,2）
#   wrapper.sh 内容：
#   #!/bin/bash
#   case $OMPI_COMM_WORLD_LOCAL_RANK in
#       0) export CUDA_VISIBLE_DEVICES=1,3,5 ;;
#       1) export CUDA_VISIBLE_DEVICES=0,2 ;;
#   esac
#   exec "$@"
mpirun -np 5 -hostfile hosts ./wrapper.sh ./nvbandwidth -t multinode_device_to_device_memcpy_read_ce
```

rank 多时可用 `-P <对数>` 随机抽样减少测量的对数（固定随机种子，结果可复现）；`--pair` 仅支持 2 rank，M2N 多卡场景不适用。

### 3.2 跨机聚合 / 特殊模式（7 个）

| key | 含义 | 备注 | 接口/算子 |
|---|---|---|---|
| `multinode_device_to_device_all_to_one_write_sm` | 所有节点 GPU → 单一设备汇聚写，报告每设备总入带宽 | 需 MPI。N2One 全网并发聚合；按目标逐一轮换；SUM_BW | `stridingMemcpyKernel` |
| `multinode_device_to_device_all_from_one_read_sm` | 单一设备 → 所有节点 GPU，报告每设备总出带宽 | 需 MPI。One2N 全网并发聚合；按源逐一轮换；SUM_BW | `stridingMemcpyKernel` |
| `multinode_device_to_device_broadcast_one_to_all_sm` | 单设备写多播内存（multicast object），一对多广播 | 需 MPI。依赖 multicast object 支持（NVSwitch/fabric 环境）；SUM_BW | `multicastCopyKernel` |
| `multinode_device_to_device_broadcast_all_to_all_sm` | 所有设备同时向各自多播内存写，全网广播 | 需 MPI。全网同时写入，最接近 all-to-all 的形态；SUM_BW | `multicastCopyKernel` |
| `multinode_bisect_write_ce` | N 卡系统同时发生 N 个拷贝：rank A 写 rank (A+N/2)%N（对分测试） | 需 MPI。N 路 1-1 拷贝同时发生（排列，非全矩阵）；偶数卡系统 | `cuMemcpyAsync` |
| `multinode_multicast_fabric_put_sm` | 每个 rank 通过 SM 向 fabric 多播内存写入（NVLink Fabric put） | 需 MPI。所有 rank 同时写多播组；SUM_BW | `multicastCopyKernel` |
| `multinode_multicast_fabric_put_tma` | 同上，TMA 实现 | 需 MPI。需 Blackwell+（TMA）；SUM_BW | `tma_memcpy_kernel` |

## 4. 单机 / 跨机支持总结

- **44 个单机测试**：仅在 `worldRank == 0 && localRank == 0` 的进程运行，测的是单机内部（主机↔GPU、GPU↔GPU）的带宽/延迟。即使以 MPI 模式启动，这些测试也只跑在 rank 0 上。
- **H2D/D2H 无跨机形态**：19 个 multinode 测试全部是 D2D 或多播变体；主机↔GPU 是每台机器的本地指标，不存在"节点 A 主机写节点 B HBM"的路径。
- **19 个 multinode 测试**：需要 MPI 多进程启动（每个 rank 绑定一块 GPU），测跨节点链路。单机构建（未定义 `MULTINODE`）时完全不编译。可用 `--pair UUID UUID` 把跨机 D2D 测试限制到指定的两块 GPU（可各在一个节点）。
- 查看当前构建实际可用的测试列表：`nvbandwidth -l`。

---

# 附录 A：CE 双流执行时序（async 拷贝模型）

bidirectional 的 CE 实现本质是 CUDA 异步拷贝——CPU 只向 stream 提交命令，拷贝由 GPU 硬件执行，最后逐个等待 stream 结束。实际比"提交两个流然后等"多了同步编排（`memcpy.cpp` 的 `doMemcpyCore`）。每个采样轮等价于：

```cpp
// 准备阶段（采样循环外）：创建 2 个非阻塞流 + 事件
cuStreamCreate(&stream0, CU_STREAM_NON_BLOCKING);   // H2D 路（被测方向）
cuStreamCreate(&stream1, CU_STREAM_NON_BLOCKING);   // D2H 路（干扰方向，buffer 为 2 倍大小）

// ── 每个采样轮 ──
memsetPattern(info);                     // 填充校验 pattern

// ① 预热拷贝（先于正式测量，两个流各一次，不计入带宽，让 Copy Engine 进入稳态）
cuMemcpyAsync(dev1, host1, stream0, WARMUP_COUNT);
cuMemcpyAsync(host2, dev2, stream1, WARMUP_COUNT);

// ② 起点对齐：stream1 等 stream0 的 start 事件，保证两方向同一时刻开始计时
cuEventRecord(startEvent0, stream0);
cuStreamWaitEvent(stream1, startEvent0);             // ★ 同时开跑
cuEventRecord(startEvent1, stream1);

// ③ 正式拷贝：异步提交，CPU 立即返回
cuMemcpyAsync(dev1, host1, size*16, stream0);        // H2D（被测方向）
cuEventRecord(endEvent0, stream0);
cuMemcpyAsync(host2, dev2, size*2*16, stream1);      // D2H（干扰方向，2 倍 buffer 保证链路饱和）
cuEventRecord(endEvent1, stream1);

// ④ CPU 等待：逐个流同步（bidirectional/USE_FIRST_BW 场景无交叉等待链）
syncStream(stream0);
syncStream(stream1);

// ⑤ 校验 + 从 GPU 事件时间线算带宽（不用 CPU 墙钟）
memcmpPattern(info);
bw0 = bytes*loopCount / elapsed(startEvent0, endEvent0);   // 只输出这一路
bw1 = bytes*loopCount / elapsed(startEvent1, endEvent1);   // 算出但不输出
```

三点说明：

1. **预热拷贝**（WARMUP_COUNT 次）先于正式测量，避免首个拷贝的冷启动开销污染测量
2. **起点事件对齐**：两个流不是提交完各跑各的，stream1 被显式挡住直到 stream0 的 start 事件触发，保证两方向真正同时开始。bidirectional 用 USE_FIRST_BW，不需要 TOTAL_BW 那套"终点归并"等待链（stream0 等 stream1 的 endEvent 再记 totalEnd）
3. **计时靠 GPU 事件不靠 CPU**：每路带宽 = 拷贝字节 × loopCount / 事件间隔，测的是 GPU 侧真实执行窗口。跨机版的 `streamBlockerBlock/Release` 是同一思想的扩展——多 rank 的流先阻塞、对齐后统一放行

---

# 附录 B：H2D-bidir 与 D2H-bidir 的执行差异

`host_to_device_bidirectional_memcpy_*` 与 `device_to_host_bidirectional_memcpy_*` 执行上**几乎完全相同**——同样的 buffer（host1/dev1 为 size，host2/dev2 为 size×2）、同样的两个拷贝、同样的并发结构，唯一区别是数组顺序互换，决定了"哪一路被报告"（`testcases.cpp` 的 `HostDeviceBidirTransfer::run`）：

```cpp
if (direction_ == Direction::HostToDevice) {
    srcBuffers = {&host1, &dev2};      // 拷贝0: host1→dev1 (H2D, size)     ← 被测
    dstBuffers = {&dev1, &host2};      // 拷贝1: dev2→host2 (D2H, size*2)   ← 干扰
} else {  // DeviceToHost
    srcBuffers = {&dev1, &host2};      // 拷贝0: dev1→host1 (D2H, size)     ← 被测
    dstBuffers = {&host1, &dev2};      // 拷贝1: host2→dev2 (H2D, size*2)   ← 干扰
}
bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);
```

具体差异只有三点，都是"视角"层面的：

1. **报告哪一路**：`MemcpyOperation` 用 `USE_FIRST_BW`，只输出数组下标 0 那路拷贝的带宽。H2D 变体报告 H→D，D2H 变体报告 D→H——物理上是同一场传输，只是挑的角度不同
2. **谁是锚定流**：起点事件对齐以 stream0 为锚（见附录 A 第②步），被测方向恰好是先记录 start 事件、干扰流等待它的那个流；两变体中锚定流角色对调。由于两个流本来就被锚点同步到同一时刻开始，这对测量结果没有实质影响
3. **context 选择的代码路径不同**：默认 `PREFER_SRC_CONTEXT` 下，H2D 变体的 src 是 host（无 context）回退到 device 执行；D2H 变体的 src 直接就是 device。但最终执行 context 都是同一块 GPU，结果无差别

结论：两者是**同一场双向传输的两个测量视角**——buffer、拷贝内容、并发结构、总流量完全相同，理论上报告的数值互为镜像（同一物理实验的两个读数）。

补充：代码原本设计让互补的一对共享测量结果（`runWithCache`，`testcase.h`），第二个测试直接取第一个的镜像值、不重跑——但缓存键不匹配（存取用了不同的键组合），实际永不命中，两个测试都会真实重跑，各测各的。

---

# 附录 C：D2D N×N 矩阵的串行与并行结构

以 4 卡为例说明 N×N 矩阵的填充过程（N×N 只是**结果矩阵的布局**，行=src，列=peer），哪些环节串行、哪些环节并行：

## read / write 测试：全程串行

```
时间线（每格是一次独立测量，一次只跑一格）：

 t1: GPU0→GPU1   （单流，16次拷贝背靠背）
 t2: GPU0→GPU2
 t3: GPU0→GPU3
 t4: GPU1→GPU0
 t5: GPU1→GPU2
 ...
 t12: GPU3→GPU2
```

- 单次测量的内部：1 个 stream、loopCount（默认 16）次拷贝排队执行、CPU 等流结束后算带宽
- 任意时刻全系统只有**一路拷贝、两个 GPU 活跃**，其余 GPU 空闲

## bidirectional 测试：对间串行 + 对内双路并发

```
时间线（每格内部有两路同时跑）：

 t1: GPU0↔GPU1   ┬─ 流0: GPU1→GPU0
                 └─ 流1: GPU0→GPU1   两路同时执行
 t2: GPU0↔GPU2   ┬─ 流0: GPU2→GPU0
                 └─ 流1: GPU0→GPU2
 ...
```

- 并发**只存在于一个矩阵格内部**：同一条链路的正反两个方向同时跑（2 个 stream + 4 块独立 buffer，src/dst 数组交叉配对形成反方向，见 `testcases.cpp` 的 `DeviceToDeviceBidirTransfer::run`），不存在跨格并行
- 对称格共用：测完 (0,1) 后，(1,0) 直接复用该次物理测量的数据填矩阵（cachedPairLoop），不再重跑——实际物理测量次数是 N×(N-1)/2，但矩阵仍是完整 N×N

## 总结对照

| 环节 | read/write | bidirectional |
|---|---|---|
| 矩阵格之间（对间） | 串行 | 串行 |
| 矩阵格内部（对内） | 单路串行（16 次迭代排队） | **双路并发**（2 流反向） |
| 对称格 (i,j)/(j,i) | 各自独立测量 | 共用一次物理测量 |
| 物理测量总次数 | N×(N-1) | N×(N-1)/2 |

即：**并行粒度永远不会超过"一对 GPU"**——nvbandwidth 的 D2D 测试从不出现"多对 GPU 同时测"的情况，N×N 矩阵是一格一格顺序测出来的。

## 控制台输出示例（数值为示意）

矩阵渲染规则：列头为 `GPU0…`（10 字符宽），每行先打行号（2 字符宽）再依次打各列值，无值处为 `N/A`；矩阵后紧跟 `SUM <testKey> <总和>`；`-v` 时追加 `MIN/MAX/AVG`；采样数 ≥3 时输出 `COEFFICIENT_OF_VARIATION` 行。

**场景一：read/write（全程串行）** —— 一个测试输出**一张** N×N 矩阵：

```
memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)
     GPU0      GPU1      GPU2      GPU3
  0    N/A       274       273       274
  1    275       N/A       274       273
  2    274       275       N/A       274
  3    275       274       273       N/A

SUM device_to_device_memcpy_read_ce 3293
```

标签中的 `->` 表示数据流向列端（read 测试 = 行 GPU 从列 GPU 读；write 测试为 `<-`）。对角线恒为 `N/A`（GPU 不对自己测）。

**场景二：bidirectional（对间串行 + 对内双路并发）** —— 一个测试输出**三张** N×N 矩阵（两路分量 + 总和）：

```
memcpy CE GPU(row) <-> GPU(column) Read1 bandwidth (GB/s)
     GPU0      GPU1      GPU2      GPU3
  0    N/A       139       138       139
  1    138       N/A       139       138
  2    139       138       N/A       139
  3    138       139       138       N/A

SUM device_to_device_bidirectional_memcpy_read_ce 1662

memcpy CE GPU(row) <-> GPU(column) Read2 bandwidth (GB/s)
     GPU0      GPU1      GPU2      GPU3
  0    N/A       138       139       138
  1    139       N/A       138       139
  2    138       139       N/A       138
  3    139       138       139       N/A

SUM device_to_device_bidirectional_memcpy_read_ce 1661

memcpy CE GPU(row) <-> GPU(column) Total bandwidth (GB/s)
     GPU0      GPU1      GPU2      GPU3
  0    N/A       277       277       277
  1    277       N/A       277       277
  2    277       277       N/A       277
  3    277       277       277       N/A

SUM device_to_device_bidirectional_memcpy_read_ce 3322
```

- `Read1` / `Read2`：同一物理测量中两路并发拷贝各自的带宽（`VECTOR_BW` 拆分）；write 测试对应标签为 `Write1` / `Write2` / `Total`
- `Total`：每格两路带宽之和，即该链路双向并发时的**双向合计带宽**（约为单向独占值的 2 倍）
- 对称格 (i,j)/(j,i) 共用一次物理测量，因此矩阵沿对角线对称

---

# 附录 D：聚合干扰类测试的执行与输出（all_to_host / host_to_all）

适用于 §2.3 前四行的 8 个测试。两个核心问题：

## 1. "多 GPU 并发"是单线程实现的

全程**只有一个 CPU 线程**，并发完全由 CUDA 异步机制实现：

```
CPU 线程（唯一）：
    为每个 GPU 建一条 non-blocking stream     // N 个 GPU → N 条流
    for 每个buffer对：
        cuMemcpyAsync(...) 到对应的流          // 逐个异步提交，立即返回
    起点事件对齐（stream1..N 等 stream0 的 start 事件）
    for 每条流：sync                          // 统一等待
```

- 提交是**顺序**的，但 `cuMemcpyAsync` 只是把命令挂进流的队列，提交完最后一个时无需等任何一个完成
- 真正的并发发生在 **GPU 硬件层**：N 条流上的拷贝由各 GPU 的 Copy Engine（或 SM kernel）同时执行
- 进程内没有第二个线程，也没有多进程（多进程并发只存在于 multinode 的 MPI 模式）

## 2. 被测 GPU 不需要指定，内部逐卡轮换

没有 CLI 参数能指定"只测哪张卡"（`--pair` 只对 D2D 类测试生效）。代码内部循环轮换：

```cpp
for (deviceId = 0; deviceId < deviceCount; deviceId++) {   // 被测对象逐卡轮换
    // 本轮：GPU[deviceId] 那一路是"测量拷贝"
    // 其余 GPU 各一路拷贝，全部只作干扰负载
    bandwidthValues.value(0, deviceId) = ...;              // USE_FIRST_BW 只取被测路
}
```

每轮把当前 GPU 的那一路作为测量对象，其余 GPU 的拷贝并发跑、只起干扰作用；全部 N 轮测完后每张卡各得一个值。

## 3. 输出样式与"如何看出谁是被测的"（数值为示意）

四类测试的输出标签各不相同（`testcases.cpp` 中 `buildOutputLabel()`）：

| 测试 | 输出标签 |
|---|---|
| `all_to_host_memcpy_ce` | `memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)` |
| `host_to_all_memcpy_ce` | `memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)` |
| `all_to_host_bidirectional_memcpy_ce` | `memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)` |
| `host_to_all_bidirectional_memcpy_ce` | `memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)`（与前一行标签相同，靠测试段落的 key 区分） |

以 `all_to_host_bidirectional_memcpy_ce` 为例（1×N 矩阵，仅一行）：

```
memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)
     GPU0      GPU1      GPU2      GPU3
  0     187       173       176       181

SUM all_to_host_bidirectional_memcpy_ce 717
```

**判读方法：矩阵的列号就是被测 GPU**。第 j 列的值 = "轮到 GPU j 作为被测对象的那一轮"的测量结果，即"在全网（含本卡反向）干扰下，GPU j 实现的带宽"。输出中没有任何显式的"measured: GPU j"标记，但这层对应关系是固定的：

- 第 0 列 → 第一轮（GPU0 被测，其余干扰）
- 第 1 列 → 第二轮（GPU1 被测，其余干扰）
- ……依此类推

双向变体的值只包含被测方向：`all_to_host_bidirectional` 第 j 列是"GPU j → 主机"方向的带宽（其 H2D 反向流与其他 GPU 的干扰都不计入数值，只影响大小）；`host_to_all_bidirectional` 第 j 列则是"主机 → GPU j"方向的带宽。

