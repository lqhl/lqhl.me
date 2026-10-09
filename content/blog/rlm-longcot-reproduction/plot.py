"""Render all article figures from analysis-data.json, without model calls.

Requires matplotlib and a Chinese font (Noto Sans CJK SC, Source Han Sans SC,
Microsoft YaHei or PingFang SC). Run from any directory:
    python plot.py

The SVGs embed glyph outlines; the exported figures need no installed fonts.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

BUNDLE = Path(__file__).resolve().parent
DATA = json.loads((BUNDLE / "analysis-data.json").read_text())
METHODS = ["base", "rlm", "rlm_tips", "codex_no_tips"]
LABELS = {"base": "单次文本", "rlm": "RLM", "rlm_tips": "RLM + tips", "codex_no_tips": "Codex + Python"}
COLORS = {"base": "#859098", "rlm": "#4478ad", "rlm_tips": "#bf8745", "codex_no_tips": "#188475"}
DOMAINS = [("math", "数学"), ("chemistry", "化学"), ("cs", "计算机"), ("logic", "逻辑"), ("chess", "棋类")]
INK, MUTED, GRID = "#253440", "#667580", "#e6ebee"
fonts = {font.name for font in font_manager.fontManager.ttflist}
font = next((name for name in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC", "Microsoft YaHei", "PingFang SC"] if name in fonts), None)
if font is None:
    raise RuntimeError("Install a Chinese font such as Noto Sans CJK SC, then rerun.")
plt.rcParams.update({
    "font.family": font, "font.size": 12, "text.color": INK,
    "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "axes.unicode_minus": False, "svg.fonttype": "path",
    "svg.hashsalt": "rlm-longcot", "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})


def header(fig, title, subtitle):
    fig.text(0.035, 0.97, title, fontsize=17, weight="bold", va="top")
    fig.text(0.035, 0.915, subtitle, fontsize=10.5, color=MUTED, va="top")


def save(fig, name):
    svg_path = BUNDLE / f"{name}.svg"
    fig.savefig(svg_path, metadata={"Date": None})
    svg_path.write_text("".join(line.rstrip() + "\n" for line in svg_path.read_text().splitlines()))
    fig.savefig(BUNDLE / f"{name}.png", dpi=250)
    plt.close(fig)
    print(f"Rendered {name}.svg / .png")


# 1. Overall accuracy and the domain distribution, on the original strict score.
fig = plt.figure(figsize=(8, 8.2))
header(fig, "Codex 与 RLM 的总体正确率接近", "同一 gpt-6-luna · 同一 507 题 · 每个条件一次运行")
ax = fig.add_axes([0.28, 0.575, 0.66, 0.25])
accuracy = [100 * DATA["methods"][m]["correct"] / DATA["methods"][m]["tasks"] for m in METHODS]
ax.barh(range(4), accuracy, color=[COLORS[m] for m in METHODS], height=0.54)
ax.set_yticks(range(4), [LABELS[m] for m in METHODS])
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
ax.tick_params(length=0, pad=7)
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
for i, (m, value) in enumerate(zip(METHODS, accuracy)):
    ax.text(value + 2, i, f"{value:.2f}%", va="center", fontsize=13, weight="bold", color=COLORS[m])
fig.text(0.035, 0.497, "分领域严格正确率", fontsize=13, weight="bold")
ax = fig.add_axes([0.22, 0.09, 0.735, 0.335])
matrix = np.array([[100 * DATA["methods"][m]["domains"][d]["correct"] /
                    DATA["methods"][m]["domains"][d]["tasks"] for m in METHODS] for d, _ in DOMAINS])
cmap = LinearSegmentedColormap.from_list("score", ["#f1f5f7", "#c1dcd8", "#137d70"])
ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(4), ["单次文本", "RLM", "RLM + tips", "Codex\n+ Python"])
ax.xaxis.tick_top()
ax.set_yticks(range(5), [f"{label}（{DATA['methods']['rlm']['domains'][d]['tasks']}）" for d, label in DOMAINS])
ax.tick_params(length=0, pad=8)
for i in range(5):
    for j in range(4):
        ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center", fontsize=12.5,
                color="white" if matrix[i, j] > 75 else INK)
ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=3)
ax.tick_params(which="minor", length=0)
fig.text(0.035, 0.025, "括号内为题数；颜色越深，正确率越高。总体按 507 题加权汇总。", fontsize=9.5, color=MUTED)
save(fig, "accuracy")

# 2. Three usage measures on the same relative scale, Codex = 1.
fig = plt.figure(figsize=(8, 7.3))
header(fig, "正确率接近，已报告输出相差 3.65 倍", "相对 Codex 的用量（Codex = 1）；条末标注实际总量")
for index, (key, title) in enumerate([("calls", "模型调用次数"), ("input_tokens", "输入 token"), ("output_tokens", "输出 token，含推理")]):
    bottom = 0.64 - index * 0.255
    ax = fig.add_axes([0.28, bottom, 0.67, 0.16])
    denom = DATA["methods"]["codex_no_tips"][key]
    methods = METHODS[1:]
    values = [DATA["methods"][m][key] / denom for m in methods]
    ax.barh(range(3), values, height=0.53, color=[COLORS[m] for m in methods])
    ax.set_yticks(range(3), [LABELS[m] for m in methods], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.9)
    ax.set_xticks([0, 1, 2, 3, 4], ["0", "1×", "2×", "3×", "4×"])
    ax.axvline(1, color="#a9b6bf", linestyle="--", linewidth=1)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=5, labelsize=10)
    ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=13)
    for i, (m, value) in enumerate(zip(methods, values)):
        actual = DATA["methods"][m][key]
        label = f"{actual:,} 次" if key == "calls" else f"{actual / 1_000_000:.2f} M"
        ax.text(value + 0.1, i, f"{value:.2f}× · {label}", va="center", fontsize=10.5, color=COLORS[m])
fig.text(0.035, 0.025, "M = 百万 token。只含最终尝试已报告用量，未包含全部预算终止及历史重试。", fontsize=9.5, color=MUTED)
save(fig, "usage")

# 3. Paired domain totals and each domain's contribution to the output gap.
fig = plt.figure(figsize=(8, 6))
header(fig, "数学贡献了 58.1% 的额外输出", "无提示 RLM 与 Codex：同一领域、同一题目；输出含推理")
ax = fig.add_axes([0.12, 0.2, 0.79, 0.57])
ordered = [("math", "数学"), ("chemistry", "化学"), ("logic", "逻辑"), ("cs", "计算机"), ("chess", "棋类")]
y = np.arange(5)
rlm = np.array([DATA["methods"]["rlm"]["domains"][d]["output_tokens"] for d, _ in ordered])
codex = np.array([DATA["methods"]["codex_no_tips"]["domains"][d]["output_tokens"] for d, _ in ordered])
ax.barh(y - 0.18, rlm / 1e6, height=0.30, color=COLORS["rlm"], label="RLM")
ax.barh(y + 0.18, codex / 1e6, height=0.30, color=COLORS["codex_no_tips"], label="Codex + Python")
ax.set_yticks(y, [label for _, label in ordered])
ax.invert_yaxis()
ax.set_xlim(0, 4.8)
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xlabel("已报告输出 token（百万）", fontsize=11, labelpad=10)
ax.tick_params(length=0, pad=7)
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.legend(loc="lower right", bbox_to_anchor=(1, 1.065), frameon=False, ncol=2, fontsize=10.5)
for i in range(5):
    for amount, offset, color in [(rlm[i], -0.18, COLORS["rlm"]), (codex[i], 0.18, COLORS["codex_no_tips"])]:
        ax.text(amount / 1e6 + 0.055, i + offset, f"{amount / 1e6:.3f}", va="center", fontsize=10, color=color)
    share = 100 * (rlm[i] - codex[i]) / (rlm.sum() - codex.sum())
    ax.text(4.75, i, f"{share:.1f}%", va="center", ha="right", fontsize=12, weight="bold")
fig.text(0.035, 0.042, "右侧为该领域占总额外输出的比例；不是该领域的浪费比例。", fontsize=9.5, color=MUTED)
save(fig, "output-by-domain")

# 4. Observed execution stops while identical ordinary text keeps appearing.
fig = plt.figure(figsize=(8, 4.7))
header(fig, "HM_easy_50：提交代码执行了，系统仍未结束", "第 15 轮执行提交代码；第 16 轮起以普通文本重复正确答案")
ax = fig.add_axes([0.13, 0.24, 0.80, 0.48])
series = DATA["hm_stall_trace"]["series"]
x = [row["turn"] for row in series]
cumulative = np.cumsum([row["code_blocks"] for row in series])
ax.axvspan(16.5, 100.5, color="#fbefe0")
ax.step([0] + x, [0] + cumulative.tolist(), where="post", color=COLORS["rlm"], linewidth=2.5)
ax.scatter([15, 16, 101], [cumulative[14], cumulative[15], cumulative[100]], color=[COLORS["rlm"], COLORS["rlm_tips"], COLORS["codex_no_tips"]], s=38, zorder=5)
ax.text(59, 8.4, "第 17–100 轮\n84 轮原文重复\n没有执行代码", ha="center", va="center", fontsize=13, color="#8b5b29", linespacing=1.5)
ax.annotate("第 15 轮\n提交未被捕获", xy=(15, cumulative[14]), xytext=(28, 18.2), fontsize=11,
            arrowprops={"arrowstyle": "-", "color": MUTED}, ha="center", color=INK)
ax.annotate("第 101 轮\n兜底返回", xy=(101, cumulative[100]), xytext=(92, 18.2), fontsize=11,
            arrowprops={"arrowstyle": "-", "color": MUTED}, ha="center", color=INK)
ax.set_xlim(0, 104)
ax.set_ylim(0, 22)
ax.set_xticks([0, 16, 40, 60, 80, 100])
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_xlabel("根模型记录轮次（不含 2 个子调用）", fontsize=11, labelpad=8)
ax.set_ylabel("累计执行的 REPL 代码块", fontsize=11)
ax.tick_params(length=0, pad=7)
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
fig.text(0.035, 0.036, "计入报错代码块；第 14 轮已将特殊 answer 对象覆盖为普通字典。", fontsize=9.5, color=MUTED)
save(fig, "submission-stall")
