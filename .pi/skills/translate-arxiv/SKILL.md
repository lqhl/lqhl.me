---
name: translate-arxiv
description: Translate an arXiv paper into a complete Chinese markdown post for this Hugo blog, keeping figures, citations and terminology working. Use when asked to translate a paper (翻译论文 / 翻译 arxiv), add a paper translation to the blog, or when a post needs citation and cross-reference links fixed.
---

# Translate an arXiv paper into a blog post

Produces `content/blog/<slug>/index.md` plus figure assets: a complete Chinese
translation with the paper's own section/figure/table numbering, an English
reference list, and working links from every citation, section reference,
figure and table to its target.

## Requirements

```bash
python3 -m pip install --break-system-packages beautifulsoup4 html2text   # once
```

`curl` is used for downloads. arXiv drops TLS connections fairly often
(`curl exit 35`); both scripts retry and are resumable, so just re-run them.

## Workflow

### 1. Fetch the source

```bash
python3 .pi/skills/translate-arxiv/scripts/fetch_arxiv.py <arxiv-url-or-id> /tmp/<id>
```

Writes `paper.html`, `paper.md` (body, with `[[FIGURE: name.svg]]` markers),
`refs.md` (cleaned bibliography) and `figs/*.svg`.

If it exits with "has no arXiv HTML version", the paper only has a PDF — use the
`pdf-reader` skill on `https://arxiv.org/pdf/<id>` instead. Never use
`fetch_content` for a full paper: it truncates at ~30k chars per slice.

### 2. Translate

Read `paper.md` and write the translation in parts (`/tmp/parts/p1.md`, `p2.md`, …),
then concatenate — a whole paper does not fit in one write. Keep the paper's
structure exactly: same section numbers, same figure/table numbers and order,
same paragraph order. Do not summarize, merge or reorder anything, and do not add
translator's notes, prefaces or "kept in English" disclaimers anywhere.

### 3. Create the page bundle

```
content/blog/<slug>/
├── index.md
└── *.svg          # copy from <outdir>/figs/
```

Front matter (TOML, this blog's convention):

```toml
+++
title = "译文 | …"
date = "YYYY-MM-DDTHH:MM:SS+08:00"
description = "一两句话，会出现在首页与列表页"
tags = ["AI", "Agent", "Software Engineering", "Paper", "Translation"]
+++
```

The title must start with the existing translation prefix `译文 | `, followed by
the Chinese title. The `date` must be in the past, otherwise Hugo silently
skips the page.

Then a source block, and the body:

```markdown
> **原文**：<English title>
>
> **作者**：Alexander Krentsel*, …, Ion Stoica（UC Berkeley）
>
> **arXiv**：<https://arxiv.org/abs/XXXX.XXXXX>

---

## 摘要
…
```

Figures use the downloaded assets and keep the English image; the caption is
translated:

```markdown
![Figure 1. Inner implementation-verification loop.](./implementation_verification_loop.svg)

> **图 1**：内层实现–验证循环。一次运行中固定 *R*、*M*、*E*，反复修订 *P*。
```

Tables become markdown tables with a bold caption line (`**表 1. 符号与定义**`).
The reference list goes under exactly `## 参考文献`, one numbered entry per line,
copied from `refs.md`.

### 4. Link everything

```bash
python3 .pi/skills/translate-arxiv/scripts/linkify.py content/blog/<slug>/index.md
hugo && python3 .pi/skills/translate-arxiv/scripts/linkify.py content/blog/<slug>/index.md \
  --check --verify-html docs/blog/<slug>/index.html
```

It adds `{#sec-2-1}` heading anchors, `<a id="fig-1">` / `<a id="tbl-1">` caption
anchors, `<a id="ref-smith-2020">` reference anchors, and rewrites
`（Smith, 2020）`, `第 2.1 节`, `图 1`, `表 2`, `附录 A` into links. It is
idempotent, refuses to write if any target is broken or any citation has no
reference entry, and prints the counts to sanity-check.

### 5. Build, review, commit

```bash
hugo                      # must succeed; docs/ is generated (gitignored)
git add content/blog/<slug> && git commit
```

Read the rendered page once for stray machine artifacts (double spaces, `RR`
instead of `R`, "Note:" fragments). One commit for the post, or one for the post
plus one for the link pass if the diff is large.

## Translation conventions

**Keep in English**: figure and table images; author names, paper titles, venue
names; the in-text citation text (`Yang et al., 2024`); product, system and
benchmark names (SWE-bench, seL4, IronFleet, gem5, Firecracker, Lean); acronyms
(API, GPU, SRE, MoE, RTL, TCB, LLM); and terms with no settled Chinese form —
`reward hacking`, `harness`, `prompt`, `agent`, `lesson`, `learning-to-defer`,
`roofline`, `bandit`, `fuzzing`, `stub`, `trace`. Symbol names stay italic:
`*R*`, `*M*`, `*E*`, `*P*`, `*I*`, `*W*`.

**Translate**: all prose, headings, table cells, figure captions, and the
"第 N 节 / 图 N / 表 N / 附录 X" references. Give the Chinese term with the
English in parentheses at first mention — 需求鸿沟（requirement gap）、
保障–修订循环（assurance–revision loop）— then use the Chinese alone.

**Punctuation and spacing**: full-width punctuation, 「」 for quoted phrases,
`（…；…）` for citation groups, and a space between CJK and Latin
(`第 2 节`, `用 *P* 表示`). One fixed translation per term for the whole post:
decide the glossary before translating and reuse it (a consistent wrong-ish term
beats three variants).

## Pitfalls (all hit in practice)

- **Figures**: the `data` attribute looks like `2609.12039v1/foo.svg`, but the
  working URL is `https://arxiv.org/html/<id>/foo.svg`. `fetch_arxiv.py` handles it.
- **Reference labels**: edited volumes are labelled by editors
  (`B. Beyer, C. Jones, … (Eds.) (2016)`), which never matches the in-text
  `Beyer et al., 2016`. `fetch_arxiv.py` normalizes the label; `linkify.py` also
  falls back to surname+year matching and reports anything it still cannot match.
- **Front matter**: never split the file on the first `+++` occurrences with a
  naive `split('+++\n')` — the opening delimiter eats the first line. Both scripts
  match `+++` lines with a regex.
- **Heading anchors** need goldmark's title attributes, on by default in Hugo.
  After building, confirm `<h2 id="sec-1">` really exists in the HTML.
- **Don't link captions or headings to themselves**, and don't link inside an
  existing link (`(?<!\[)` guards this).
- **Don't add table CSS** to `static/css/custom.css` to make wide tables pretty —
  the theme styles all posts alike.
- **Commit the page bundle**, not a loose `.md` file: images live next to
  `index.md`.

## Checklist

- [ ] every section, figure, table, appendix and reference of the original is present
- [ ] no translator's notes or meta commentary
- [ ] `hugo` builds; the post appears in `docs/blog/<slug>/`
- [ ] `linkify.py --check --verify-html` reports 0 broken targets and 0 unmatched citations
- [ ] images render; a spot check of one dense table looks readable
