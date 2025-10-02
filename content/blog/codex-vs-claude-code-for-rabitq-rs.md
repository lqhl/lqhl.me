+++
title = "从 RaBitQ Rust 移植看 AI 编程工具的实力差距：Claude Code vs Codex"
date = "2025-10-02T21:40:00+08:00"
description = "通过 RaBitQ 向量量化算法从 C++ 到 Rust 的实际移植项目，对比 Claude Code 和 Codex 两大 AI 编程工具的真实表现。Claude Code 在一天内将性能从慢 10 倍优化到接近 C++ 水平，展现出深度代码理解、主动问题诊断和系统性优化能力。"
tags = ["AI", "Claude", "Codex", "LLM", "AI Coding"]
+++

## 项目背景：为什么选择 RaBitQ？

RaBitQ 是 NTU VectorDB 团队开发的向量量化算法，能够在 1-bit per dimension 的极限压缩下仍保持高精度，并提供理论误差界保证。这在向量数据库领域是一个重要突破。

该算法的核心优势包括：在 1-bit 量化下就能达到可用精度；使用 4-bit、5-bit、7-bit 分别可实现 90%、95%、99% 的召回率；支持基于位运算的快速距离估计。

RaBitQ 已被多个工业系统采用，包括 Milvus、Faiss、Elasticsearch（命名为 BBQ）、Lucene（命名为 BBQ）、CockroachDB、Volcengine OpenSearch 等。这些应用覆盖了从开源到商业的主流向量数据库。

我决定将 [RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library) 的 C++ 实现移植到 Rust，为 Rust 生态提供这一先进算法。项目地址：[lqhl/rabitq-rs](https://github.com/lqhl/rabitq-rs)

这次移植让我意外地对比了两个 AI 编程工具的真实能力。

## 第一阶段：Codex 的功能实现

使用 OpenAI Codex 完成了基础工作：

- 核心量化算法的 Rust 实现
- IVF 索引结构
- 基本搜索功能和数据集工具
- 后续的 k-means 优化和进度显示

代码能够编译运行，测试通过。看起来是个成功的移植。

### 性能现实

然而实际测试暴露了严重问题：**Rust 版本比 C++ 原版慢了 10 倍以上**。

对于一个以性能为核心价值的量化算法，这样的性能是不可接受的。我多次要求 Codex 优化，得到的回复是"已经完全按照 C++ 版本实现"。

检查代码后发现，Codex 做的是语法级别的转换：

```rust
fn quantize_vector(data: &[f32], centroid: &[f32]) -> Vec<u8> {
    let mut codes = vec![0u8; (data.len() + 7) / 8];
    for i in 0..data.len() {
        let rotated = data[i] - centroid[i];
        if rotated > 0.0 {
            codes[i / 8] |= 1 << (i % 8);
        }
    }
    codes
}
```

这段代码功能正确，但完全没有考虑 C++ 版本中的性能优化技巧。

## 第二阶段：Claude Code 的系统性优化

切换到 Claude Code 后，情况完全不同。

### 精准诊断

Claude Code 首先给出了完整的性能分析，指出 Rust 版本缺失的四个关键优化：

1. **Fast Hadamard Transform (FHT) 旋转**：C++ 使用 O(dim log dim) 的 FHT，Rust 用的是 O(dim²) 的矩阵乘法
2. **faster_config 模式**：C++ 可以预计算常数缩放因子，Rust 每个向量都要优化
3. **查询预计算**：C++ 缓存了查询相关常数，Rust 在重复计算
4. **存储格式**：C++ 只存 FHT 状态，Rust 保存了完整旋转矩阵

这些诊断都准确地对应了 C++ 实现的核心优化。

### 技术实现

#### 1. Fast Hadamard Transform (commit 2039847)

实现了与 C++ 一致的 FhtKacRotator：

```rust
pub struct FhtKacRotator {
    signs: Vec<BitVec>,           // 4 轮符号翻转
    kac_walks: Vec<Vec<usize>>,   // Kac 随机游走
    dim: usize,
    padded_dim: usize,            // 对齐到 64 的倍数
}

impl FhtKacRotator {
    fn apply(&self, data: &mut [f32]) {
        for round in 0..4 {
            self.fht_inplace(data);
            self.apply_signs(data, round);
            self.apply_kac_walk(data, round);
        }
    }
}
```

这不是简单的代码翻译，而是理解了 FHT 在量化中的作用：通过快速的正交变换改善数据分布，使量化误差更均匀。

**效果**：旋转速度提升 10-100 倍，索引构建时间大幅缩短。

#### 2. faster_config 模式 (commit 04c477a, bf08722)

添加了快速训练选项：

```rust
pub struct RabitqConfig {
    pub dim: usize,
    pub total_bits: usize,
    pub t: Option<f32>,  // None=动态优化，Some(t)=固定常数
}

impl RabitqConfig {
    pub fn faster(dim: usize, total_bits: usize, seed: u64) -> Self {
        let t_const = compute_const_scaling_factor(dim, seed);
        Self { dim, total_bits, t: Some(t_const) }
    }
}
```

通过采样 100 个随机向量预计算一个通用的缩放因子，避免了每个向量的独立优化。

**效果**：训练速度提升 100-500 倍，准确度损失小于 1%。

#### 3. 查询预计算 (commit 04c477a)

引入了查询预计算结构：

```rust
pub struct QueryPrecomputed {
    query_norm: f32,
    k1x_sum_q: f32,
    kbx_sum_q: f32,
    binary_scale: f32,
}
```

在搜索开始时一次性计算这些常数，后续所有候选向量共享这些值。典型查询会检索 `nprobe × cluster_size` 个向量，这个优化消除了大量重复计算。

**效果**：查询性能显著提升。

#### 4. 存储优化 (commit 2039847)

只保存 FHT 的紧凑状态：

- 原方案：`dim × dim × 4 bytes`（GIST-960 需要 3.6MB）
- 新方案：`4 × (padded_dim/8 + padded_dim×2) bytes`（GIST-960 只需 8.5KB）

**效果**：存储减少 99.7%，加载速度大幅提升。

### 优化成果

经过这些系统性改进，Rust 版本的性能**已经接近 C++ 原版水平**。

具体改进：

- 索引构建从慢 10+ 倍优化到接近 C++ 性能
- 查询性能大幅提升
- 存储开销从 +3.6MB 降至 +8.5KB
- 代码质量达到生产就绪标准

### 开发效率

**Codex**：数天时间完成功能，但性能不达标

**Claude Code**：一天内完成诊断和优化，产出 6 个高质量 commits：

```
2039847 - Align with C++ RaBitQ (FHT + storage)
04c477a - Add faster_config and query precomputation  
bf08722 - Add Faster Config Support for training
a391383 - Simplify Quantizer API
94b8ab9 - Apply Cargo Fmt
45531c1 - Fix Clippy Warnings
```

每个 commit 都有明确目标和完整实现，不是试错式的修补。

## 核心差异分析

### 代码理解的深度

**Codex**：看到 C++ 的矩阵乘法，翻译成 Rust 的矩阵乘法

```cpp
Matrix R = gram_schmidt(random_matrix(dim, dim));
```

↓

```rust
let r = gram_schmidt(random_matrix(dim, dim));
```

**Claude Code**：理解旋转的真正目的，选择更优方案

```rust
// 知道 FHT 可以达到相同的统计性质，且快 100 倍
let rotator = FhtKacRotator::new(dim, seed);
```

### 优化意识

**Codex**：测试通过即认为任务完成，对性能问题缺乏主动意识

**Claude Code**：主动识别性能瓶颈。即使我只说"比 C++ 慢"，它也能精准定位到 FHT、faster_config、查询预计算和存储格式这四个关键点。

### 工程思维

**Codex**：关注功能实现

**Claude Code**：同时考虑性能、存储、可维护性。每次优化都明确说明：

- 性能提升幅度（10-100x、100-500x）
- 准确度影响（<1%）
- 存储开销（99.7% 减少）

这是工程师的思维方式，不只是"让代码跑起来"。

### 领域知识

这个项目涉及的专业知识包括：

- 高维向量的量化理论（Johnson-Lindenstrauss 引理）
- Fast Hadamard Transform 及其在随机投影中的应用
- SIMD 优化和缓存友好的数据布局
- 向量数据库的索引结构（IVF、HNSW）

Claude Code 在所有这些领域都展现出专家级的理解。它不只是知道代码怎么写，而是理解算法为什么这样设计、每个优化的理论基础是什么。

## 实用建议

基于这次经验，针对不同场景的建议：

**性能关键的系统代码**：

- 推荐 Claude Code
- 适用场景：数据库内核、编译器、量化算法、图形引擎
- 原因：需要深度优化和算法级理解

**功能开发和快速原型**：

- Codex 或 Claude Code 都可以
- 适用场景：Web 服务、业务逻辑、工具脚本
- 原因：功能正确性比极致性能更重要

**学习和实验**：

- 任何工具都可以尝试
- 注意：需要人工审查，理解原理比代码更重要

## 结论：Claude Code 是最强的 AI 编程工具

基于 RaBitQ Rust 移植的实战经验：

| 维度     | Codex            | Claude Code   |
|----------|------------------|---------------|
| 开发时间 | 数天             | 一天内        |
| 性能表现 | 比 C++ 慢 10+ 倍 | 接近 C++ 水平 |
| 代码质量 | 功能可用         | 生产就绪      |
| 问题诊断 | 需要明确指导     | 主动识别      |

**Claude Code 就是目前最强的 AI 编程产品。**

理由：

1. **深度理解**：不做表面翻译，而是理解算法本质和性能特征
2. **主动优化**：在短时间内识别并解决了四个关键性能问题
3. **专家知识**：掌握量化理论、FHT、SIMD 等专业领域知识
4. **工程质量**：产出的代码达到生产级标准，包括测试、文档、最佳实践

此外，我也通过 API 方式尝试了其他 AI 模型（包括 Kimi K2 和 GLM-4.6），效果都不如 Codex，更远远落后于原生 Claude Code。

**对于需要高性能、高质量的严肃项目，Claude Code 是唯一的选择。**

---

**项目信息**：

- C++ 原版：[VectorDB-NTU/RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library)
- Rust 移植：[lqhl/rabitq-rs](https://github.com/lqhl/rabitq-rs)
- 论文：[SIGMOD 2025] Practical and Asymptotically Optimal Quantization
