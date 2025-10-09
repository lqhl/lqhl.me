+++
title = "从 SPTAG 到 SPANN，再到 SPFresh：亿级向量检索的技术脉络与细节"
date = "2025-10-09T22:24:00+08:00"
description = "本文深入解析 SPTAG、SPANN 和 SPFresh 三种向量检索技术的演进脉络，从树图混合索引到磁盘分区存储，再到增量更新与自平衡机制，揭示亿级向量检索的技术细节与优化策略。"
tags = ["Vector-Search","Database","System","ANN",]
+++

## 1. SPTAG：树 + 图的内存导航器

**SPTAG（Space Partition Tree And Graph）** 是微软开源的近邻检索库，提供两套混合索引：
**KDT**（kd-tree + RNG）和 **BKT**（Balanced k-means tree + RNG）。KDT 构建轻、BKT 在高维更准；两者都用 **RNG（Relative Neighborhood Graph, 相对邻域图）** 增强连通性，并支持 L2 / Cosine。搜索时**先在树上找种子，再在图上迭代扩展**。

### 1.1 kd-tree 在 ANN 里的用法

kd-tree 递归用超平面划分空间。常见做法：在方差最大的维度取**中位数**切分，使叶子容量受控。查询时自顶向下选“更可能一侧”，同时以堆维护候选并在必要时回溯另一侧。优点是**快速收敛、构建快**；缺点是高维时“距离上界”估计保守，容易漏掉跨超平面的近邻。

### 1.2 Balanced k-means tree (BKT)

BKT 用“**均衡的 k-means**”做层次聚类：把点集按 k-means 分成 k 簇，并**加平衡约束**让每簇规模接近，然后对每个子簇递归，直至叶结点大小达到阈值。均衡性让每条路径“代价相近”，在**高维更鲁棒**。SPTAG 的 BKT 正是基于这种思想实现的内存导航树。

### 1.3 RNG（相对邻域图）到底是什么

RNG 是稀疏化的邻接图：对任意两点 p、q，仅当不存在第三点 r 让
max(d(p, r), d(q, r)) < d(p, q) 时，p–q 才连边（直观：p、q 的“透镜”中没有更近的 r）。SPTAG 在实践中通常**先构 kNN 图再按 RNG 规则稀疏化/修剪**，得到**边更少但可达性好的**邻域图。这样既保留“局部可跳跃”，又控制内存与遍历分支。

### 1.4 SPTAG 的检索流程（简版伪代码）

```
种子 = 树检索(BKT/KDT, q, 返回若干叶/中心)
候选队列 = 种子
已访问 = {}
while 未达到 MaxCheck 且 候选非空:
    x = 候选队列.pop_best()
    for 邻居 y in RNG[x]:
        if y 未访问:
            计算 d(q, y)，压入候选队列
            标记访问
返回 Top-k
```

树给**方向感**，图保证**局部精修**。这也是后面 SPANN 和 SPFresh 的“内存层导航”基石。

示意图（树上收敛 + 图内细化）：

```
          [Root]
         /  |   \
      ...  ...  ...
       |         |
   [Region A]  [Region B]
      |             |
   RNG(A)        RNG(B)
```

## 2. SPANN：质心在内存、向量在磁盘的混合索引

**SPANN** 把“内存导航 + 磁盘分区”的两层结构工程化：
在**内存**只放**Posting 质心的 SPTAG 索引**；在**磁盘**落一大批**均衡的 Posting 列表**。查询先在内存里找最近若干质心，再并行拉取对应 postings 做精排。这样可以**少访盘、尾延迟稳定**。

### 2.1 为什么“一定要均衡”

Posting 长度如果差异大，查询命中长 posting 时 I/O 与计算会暴涨，造成尾延迟的强抖动。SPANN 用**多约束均衡聚类**把 |X| 个向量划成 N 个 posting，同时最小化 posting 长度方差；并用**层次化**来把复杂度从 O(|X|·m·N) 降到 O(|X|·m·k·logₖN)。

示意图（两层结构）：

```
内存：SPTAG(质心图) —— 找最近的若干质心
  |                                   ^
  v                                   |
磁盘：Posting 列表  P(c1), P(c2), ...  ———— 并行取回做精排
```

### 2.2“边界向量”如何判定与复制（Closure Assignment）

只搜索 K 个 posting 会漏掉“在边界上的真近邻”。SPANN 在**最终一层**引入**闭包式多簇赋值**：
把向量 x 赋给多个**距离几乎相同**的最近簇。形式化地，若

```
Dist(x, c_ij) ≤ (1 + ε1) · Dist(x, c_i1)     且   Dist(x, c_i1) ≤ Dist(x, c_i2) ≤ … ≤ Dist(x, c_iK)
```

则 x 同时属于这些簇（只复制“边界向量”，靠近簇心的点不复制）。这样**在任一近邻 posting 被搜索时**，这些边界向量都有被召回的机会，**用少量冗余换高召回**。

> 这就是“什么样的向量是边界向量”：满足上式阈值条件（与最近两个或几个簇**距离几乎相同**）的点。

### 2.3 复制去冗：代表性复制 + RNG 规则

边界复制容易让几个**非常接近**的 posting 含有大量相同向量，造成**盘读浪费**。SPANN 进一步用**RNG 规则**选择**代表性簇**来挂载副本，避免“复制给过近、方向相同的簇”。论文给了一个简化判定：
**跳过**簇 (i_j) ，如果 `Dist(c_ij, x) > Dist(c_{i(j-1)}, c_ij)`；直觉是**只保留“方向差异大”的多个簇**，减少相邻 posting 的内容重叠。

### 2.4 查询时的“按需扩展”与质心替换

* **动态剪枝**：不是固定搜 K 个 posting，而是当 `Dist(q, c_ij) ≤ (1+ε2)·Dist(q, c_i1)` 时才把该 posting 列入“必搜”，从而减少不必要 I/O。
* **质心代表替换**：用**最靠近质心的真实向量**替代几何中心做导航，**把“找质心”的开销转化为“算真实候选距离”**，更划算。

## 3. SPFresh：在 SPANN 之上的在线增量更新与自平衡

SPANN 是静态构建。真实线上数据分布会漂移，posting 逐渐失衡。**SPFresh** 在 SPANN 上叠了一层**轻量增量重平衡协议 LIRE**，把“重建”改成**小步快跑的局部自愈**，并把存储层拉到用户态做可预测的 I/O。

示意图（SPFresh 端到端）：

```
Query → 质心导航(SPTAG-BKT) → 选 posting → 盘上扫描
            ^                         |
            |      Insert/Delete      |
         Updater  ────────────────────+
            |                         |
            v                         |
     Local Rebuilder (LIRE: Split/Merge/Reassign)
            |
            v
   存储：SPDK(直通NVMe) 或 RocksDB(可选)
```

### 3.1 LIRE：只动必要的局部，保持 NPA

**五类操作**：Insert、Delete、Split、Merge、Reassign。
**不变量 NPA（Nearest Partition Assignment）**：每个向量应归属于**最近**的 posting。Split/Merge 后，只对**可能违反 NPA 的近邻 posting**触发 Reassign，工作集极小，且**有限步收敛**。在 100 天模拟里，只有约 **0.4%** 插入触发重平衡；平均评估 **~5094** 个向量，仅 **~79** 个实际被重分配；Merge 频率约 **0.1%**。这解释了**P99.9 长期稳定**。

### 3.2 Version Map：并发友好的逻辑时间

每个向量维护 1 字节元数据：**7 bit 版本 + 1 bit 删除**。Insert/Reassign 版本自增（模 128 循环），Delete 置位；查询过滤旧版本与已删项；后台延迟 GC。用 CAS 保证原子更新，使写路径**append-only** 且线程安全（来自论文系统设计与实现章节描述）。

### 3.3 存储层：SPDK 为主，RocksDB 为备

* **SPDKIO**：用户态直通 NVMe，提供 `GET/APPEND/PUT` 等原子接口，降低写放大，**把瓶颈推到 SSD IOPS**，尾延迟可控。
* **RocksDBIO（可选）**：仓库同时实现了基于 RocksDB 的存储控制器，接口一致、部署门槛更低（有自定义 MergeOperator、blob/缓存/压缩等配置），但要承担 LSM 风格的读/写放大与 compaction 抖动。

### 3.4 为什么 DiskANN 重建期尾延迟会飙升

论文在 1%/day 更新的长跑实验里观测到：**DiskANN 的 P99.9 在全局重建（global rebuild / streamingMerge）期间会陡升到 20ms 以上**，即便启用了 10ms 的“硬超时”也挡不住尖峰。原因很直接：

1. **全局图重建**需要遍历并重连大量节点/边，计算密集；
2. 重建与查询**竞争 CPU/内存/I/O**，搜索线程会被阻塞或饥饿；
3. 即便采用“删除旧边 + 用已删邻居的邻域补边”的轻量策略，也难阻止质量随时间下降，从而引发周期性重建与**尾延迟锯齿**。

**SPFresh 的不同**：LIRE 只在局部做 Split/Merge/Reassign，无全局锁与大规模重算，**P99.9 平稳 ~4ms**，对比基线系统平均 **2.41× 更低**。

对比小图（资源抢夺 vs 局部自愈）：

```
DiskANN： [Rebuild] ======== 抢锁/CPU/IO ========> [Search]
SPFresh： [Local Rebuilder] -- 小批局部 --        [Search]
```

## 4. 最后把三者拼起来看

* **SPTAG（2018+）**：用 KDT/BKT 给搜索“定向”，用 RNG 保持可达性与稀疏度；在线插删、分布式服务化工具链完整。
* **SPANN（2021）**：分层（内存质心 + 磁盘 postings）、**均衡聚类**控制 tail、**边界复制**配合 RNG 代表性选择，少量冗余换高召回；查询时按需扩展，减少无效 I/O。
* **SPFresh（2023）**：在 SPANN 之上用 **LIRE** 做**增量自平衡**，配合 SPDK/RocksDB 存储，避免重建造成的长尾与资源峰值；在 1B 压测里能把瓶颈推到 NVMe IOPS，**稳定的 P99.9 与召回**并存。

总览数据流（职责分层）：

```
[SPTAG-BKT/KDT] —— 选质心 ——> [SPANN postings on disk]
        ^                               |
        |                           LIRE 局部自愈
        +—— Updater —— Local Rebuilder ——+
                          |
                     [SPDK/RocksDB]
```
