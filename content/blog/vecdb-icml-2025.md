+++
title = "VecDB@ICML2025 里有趣的论文"
date = "2025-10-16T13:00:00+08:00"
description = "整理 VecDB@ICML2025 里有趣的论文，主要是关于 vector database 和 approximate nearest neighbor (ANN) 的。"
tags = ["Vector-Database", "ANN", "Research", "RAG"]
+++

> https://vecdb-ws.github.io/icml2025/index.html

VecDB 是 ICML 里针对 vector database 的 workshop，2025 年是第一届。除了 vector database 和 approximate nearest neighbor (ANN)，也有一些是关于 RAG 和 LLM 推理的。

## ANN

跟 ANN 相关的好玩工作。

### Best Paper Award: [A Bi-metric Framework for Efficient Nearest Neighbor Search](https://openreview.net/forum?id=116kjHx1MF), Haike Xu, Piotr Indyk, Sandeep Silwal

传统的 "retrieve-then-rerank" 方案使用 cheap metric d (L2/Cosine) 召回 k 个候选，再用 expensive metric D (cross-encoder) 重新排序。其缺陷在于：

1. 如果 d 与 D 的近似比 C > 1，则最终只能保证 C 倍近似；
2. 必须对所有 k 个候选逐一计算 D，计算量线性随 k 增长。

论文提出的**双度量（bi-metric）框架**则反其道而行：

- 索引阶段只用便宜的 d 构建数据结构（如 DiskANN 或 Cover Tree）；
- 查询阶段主要用 D 评估，但只需对少量点调用。这样可在**保持 D 的准确性**的同时，显著降低对 D 的调用次数。

形式化地，若 d 与 D 满足 d(x,y) ≤ D(x,y) ≤ C·d(x,y)，则构建于 d 的索引在查询时只需 Õ(Q(ε/C, λ_d)) 次 D 评估即可找到 (1 + ε) 近邻。

应用在 DiskANN 算法，索引阶段不变。查询阶段分两步：

1. **阶段 1 – 用 d 进行粗搜索**
    - 从随机或中心节点出发，运行 DiskANN 贪心搜索；
    - 每一步选择邻居中与查询向量 q 距离最小的（按 d 计算）；
    - 维护一个候选队列 A；
    - 当访问过的节点数达到 Q/2（论文默认预算的一半）时停止；
    - 得到一批候选顶点 S_d​。
    这一阶段只调用 cheap 度量 ddd，代价极低。
2. **阶段 2 – 用 D 在同一图上继续搜索**
    - 以 S_d​ 作为起点；
    - 重启 DiskANN 的贪心搜索算法，但此时所有比较都用昂贵的度量 D；
    - 继续扩展队列 A，直到预算 Q 次 D 调用耗尽；
    - 返回目前距离 D(q,p) 最小的前 k 个点。
可以理解为：**d** 决定起步方向，**D** 决定最后的路径收敛。

在 bi-metric 方法中，需进行 200 次 D 计算（Q=200），即可达到先进行近似最近邻搜索再通过 800 次 D 计算进行重排的同等效果。

bi-metric 与推荐系统中的二向箔概念颇为相似，一个可行的思路是：在二向箔的 HNSW 中先采用 L2 或 IP 距离进行初步搜索，随后再切换至基于 NN 的度量计算方法进行精确匹配。

### [α-Reachable Graphs for Multi-vector Nearest Neighbor Search](https://openreview.net/forum?id=v8jSxLHEE9), Siddharth Gollapudi, Ravishankar Krishnaswamy, Ben Landrum, Nikhil Rao, Kirankumar Shiragur, Sandeep Silwal, Harsh Wardhan

α-Reachable Graphs 是一种理论上有保证、实践上高效的**多向量最近邻检索方法**，将经典的 DiskANN 图结构推广到非度量（non-metric）相似度函数，例如 Chamfer 距离——这是 ColBERT 等多向量检索模型的核心相似度度量。简单来说，他们证明了：**只要距离函数满足弱三角不等式，DiskANN 仍然能跑得快、搜得准。**

作者直接把 **DiskANN** 的距离函数替换成 Chamfer，相当于 **不改算法，只改距离** 的方式验证理论。仅替换距离函数的 **Chamfer-DiskANN** 已能超越复杂变换 - 编码类方法（MUVERA），并在理论上得到收敛保证。

这篇论文的思路与最佳论文 **bi-metric** 非常相似。不同之处在于，bi-metric 方法在建图和搜索的前半部分都采用了成本较低的度量方式（如 L2 或 IP），而 α-Reachable Graphs 则始终使用计算代价更高的 Chamfer 距离。相比之下，bi-metric 方法适用范围更广且成本更低。两种方法都经过了严格的理论分析。

### [Down with the Hierarchy: The 'H' in HNSW Stands for "Hubs"](https://openreview.net/forum?id=OJwITuuU3h), Munyampirwa, Vihan Lakshman, Benjamin Coleman

移除 HNSW 中的层级结构（Hierarchical）对性能影响甚微，反而能降低存储成本。这是因为高维向量构成的图（即 HNSW 的最底层）天然存在大量枢纽节点 (hub)，这些节点足以替代原有的层级结构。HNSW 相较于 NSW 的优势主要源于其搜索策略。作者开源了 [FlatNav](https://github.com/BlaiseMuhirwa/flatnav)，该项目仅保留了 HNSW 的最底层结构。

这一改进方案在生产环境中应易于验证效果。然而，对于推荐系统场景，该方法可能并不完全适用，因为推荐场景中常存在较多数据孤岛？

### The RaBitQ Library](https://openreview.net/forum?id=OeZHhOsFir), Jianyang Gao, Yutong Gou, Yuexuan Xu, Jifan Shi, Zhonghao Yang, Cheng Long

ANN 届新的超级巨星 RaBitQ 团队为他们新的 [开源库 RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library) 写了一篇论文。

### [Scaling Laws for Nearest Neighbor Search](https://openreview.net/forum?id=nXAlM7xci6), Philip Sun, Felix Chern, Yaroslav Akhremtsev, Ruiqi Guo, David Simcha, Sanjiv Kumar

作者 Philip Sun 和 Ruiqi Guo 来自 ANN 届的老牌超级巨星 ScaNN 团队，不得不看看。

核心主张：**没有 " 最优 " 的近邻算法，只有特定场景下的最优平衡。**

作者在 ScaNN、DiskANN、Faiss 等系统上做大规模实验，发现一个稳定的**三次根规律（cube-root law）**：

$$
 C(N) \propto N^{1/3}
$$

含义：

- 当数据规模增加 1000 倍，查询代价仅增加约 10 倍。
- 表明 ANN 的剪枝效率随规模提升而增强。

在 TPU 等高带宽硬件上，暴力搜索因**算术密集度高、内存访问规律**而表现突出

- TPU-KNN 在 200 万向量时的 QPS/TCO 优于 ScaNN 与 HNSW。
- 对数千万规模仍具竞争力。因此，在**数据量中等或索引频繁重建**的场景中，Brute Force 可能是最优选择。

在分布式场景下：

- 分区式（树状）结构 → 可分布、低通信、可扩展。
- 图结构 → 跨边过多、延迟大、难扩展。

### [DistributedANN: Efficient Scaling of a Single DiskANN Graph Across Thousands of Computers](https://openreview.net/forum?id=6AEsfCLRm3), Philip Adams, Menghao Li, Shi Zhang, Li Tan, Qi Chen, Mingqin Li, Zengzhong Li, Knut Magne Risvik, Harsha Vardhan simhadri

这篇论文展示了如何将 DiskANN 扩展到**数千台机器**的规模，以实现在线检索**高达 500 亿级别向量**的能力。

实现方法如下：

1. 将 DiskANN 迁移到分布式键值存储系统（例如 TiKV 这类系统）上运行。
2. 原本 DiskANN 将每个向量的乘积量化（PQ）结果存储在内存中，现在改为存储在每个图节点中。这意味着每个 PQ 向量需要存储多份，每一条入边都需保存一份。
3. 引入了一个纯内存的头部索引，用于存放 Vamana 图的顶层节点，以便快速定位 beam search 的起始点。

前一篇 Google 的论文表示尚未见到性能优异的分布式图结构向量索引方案，而微软的这个工作就当场打脸了。不过，微软实际上是通过显著增加存储冗余（即多份存储 PQ 向量）的方式，才解决了图索引难以分布式部署的难题。

### [IVF² Index: Fusing Classic and Spatial Inverted Indices for Fast Filtered ANNS](https://openreview.net/forum?id=kXw8E3xT7O), Ben Landrum, Magdalen Dobson Manohar, Mazin Karjikar, Laxman Dhulipala

论文的关键观察是：真实世界中的标签频率往往呈 **幂律（power-law）分布**。

也就是说，大多数标签对应的数据点很少，只有极少数 " 热门标签 " 包含大量向量。

基于这一现象，IVF² 采用**混合式索引策略**：

- 对**小标签（少量向量）**，直接用**简单倒排表（flat posting list）**暴力搜索；
- 对**大标签（热门标签）**，建立更复杂的**空间索引**结构（IVF + Vamana graph）；
- 同时使用**位向量（bit vector）**和**物化交集（materialized joins）**来高效处理 **AND 查询**。

这种 " 倒排 + 空间索引 " 的融合方式，即为 **IVF²（Inverted File × Inverted File）**。

不过，我个人更倾向于使用与查询条件无关 (predicate-agnostic) 的索引（如 ACORN/NaviX），对于这种需要针对每个标签单独处理的方式兴趣不大。

### [Entropy-Based Dynamic Hybrid Retrieval for Adaptive Query Weighting in RAG Pipelines](https://openreview.net/forum?id=bwGaZOVo0c), John Richard Perez, James Yuncheng Zhou, Radley Le, Alexander Menchtchikov, Ryan Lagasse

在进行稠密向量与稀疏向量的混合搜索时，最终得分通常由公式 $s = w_s \cdot s_{\text{sparse}} + w_d \cdot s_{\text{dense}}$ 计算得出。通常情况下，权重系数满足 $w_d = 1 - w_s$ 的关系。

这篇论文提出了一种基于信息论的方法，能够根据稠密/稀疏向量的确定性（熵）自适应地调整参数 $w_s$ 和 $w_d$。

1. 一次性检索：用 BM25 与 FAISS 各取 top-$k$ 结果（例如 $k=5$）。
2. 计算熵：对每种检索的打分归一化为概率分布：

$$ p_i = \frac{s_i}{\sum_j s_j} $$

计算标准 Shannon 熵：

$$ H = -\sum_i p_i \log p_i $$

归一化为：

$$ \hat{H} = \frac{H}{\log k} \in [0, 1] $$

解释：熵高 → 不确定性大 → 该检索不应占大权重。
3. 迭代更新权重：

$$ w_s^{(t+1)} = \frac{1 - \hat{H}_{sparse}}{(1 - \hat{H}_{sparse}) + (1 - \hat{H}_{dense})} $$

$$ w_d^{(t+1)} = 1 - w_s^{(t+1)} $$

迭代直到权重变化满足：

$$ |\Delta w_s| < \epsilon $$
或达到最大迭代次数 $n$。
4. 融合排序：

$$ S_i^{(*)} = w_s^{(*)} \cdot S_{sparse,i} + w_d^{(*)} \cdot S_{dense,i} $$

以 $S^{(*)}$ 排序得到最终 top-$k$ 结果。

这样，系统能在无需重新检索的情况下，通过熵调整完成 " 自适应融合 "。

## 其他

### [Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs](https://openreview.net/forum?id=1iBrBNngRh), Ryan Synk, Monte Hoover, John Kirchenbauer, Neel Jain, Alex Stein, Manli Shu, Josue Melendez Sanchez, Ramani Duraiswami, Tom Goldstein

**通过在 LLM 推理中使用向量数据库，解决长文本的性能问题！**

LLM 的注意力是稀疏的，在每一步生成时，可以只选择与当前查询最相关的 k 个 token 参与注意力计算，而不是全部 N 个。在内存里维护一个 HNSW 索引，不断的把新生成的 token 的 key 向量塞进去，下次生成的时候查询 top K 的 key 向量，拿到对应的 value 向量，给回 GPU 做推理。

### [FrugalRAG: Learning to retrieve and reason for multi-hop QA](https://openreview.net/forum?id=ZHHhlQbXnc), Abhinav Java, Srivathsan Koundinyan, Nagarajan Natarajan, Amit Sharma

1. **大规模微调并非必要**。一个精心设计的 ReAct 提示框架即可超过很多最新方法
2. **引入 " 节俭性（Frugality）" 指标**，关注模型在推理时的检索次数；
3. **通过少量数据（仅 1000 条样本）结合 RL 微调**，在保持准确性的同时将检索次数减半。

### [ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via Reinforcement Learning](https://openreview.net/forum?id=1ZxNjIVGdO), Changtai Zhu, Siyin Wang, Ruijun Feng, Kai Song, Xipeng Qiu

第一个无需外部监督的查询改写框架。通过强化学习直接利用向量数据库的检索信号优化查询改写，消除对外部改写监督的依赖。
