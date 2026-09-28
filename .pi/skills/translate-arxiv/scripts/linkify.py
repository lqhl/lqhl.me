#!/usr/bin/env python3
"""Add cross-reference anchors and links to a translated paper post.

Usage:
    python3 linkify.py <post.md> [--check] [--verify-html <built.html>]

Conventions the post must follow (see the translate-arxiv skill):
  * reference list under a `## 参考文献` / `## References` heading,
    one entry per line: `N. Label (Year) rest...`
  * figure captions as `> **图 N**：...`, table captions as `**表 N. ...**`
  * numbered headings `## N.` / `### N.M.`, appendices `## 附录 X：`

What it does:
  1. heading anchors   `## 2.1. ...`            -> `{#sec-2-1}`
  2. figure anchors    `> **图 1**：...`         -> `> <a id="fig-1"></a>**图 1**：...`
  3. table anchors     `**表 1. ...**`          -> `<a id="tbl-1"></a>**表 1. ...**`
  4. reference anchors `1. Smith (2020) ...`    -> `1. <a id="ref-smith-2020"></a>Smith (2020) ...`
  5. in-text links     `（Smith, 2020）`         -> `（[Smith, 2020](#ref-smith-2020)）`
                       `第 2.1 节`, `图 1`, `表 2`, `附录 A`

Idempotent: already-anchored and already-linked text is left alone.
"""

import argparse
import re
import sys
from pathlib import Path

REFS_HEADING = re.compile(r"^##\s+(参考文献|References)\s*$", re.M)
FRONTMATTER = re.compile(r"^\+{3}\s*$", re.M)
CAPTION_FIG = re.compile(r"^> \*\*图 (\d+)\*\*[：:]")
CAPTION_TBL = re.compile(r"^\*\*表 (\d+)\.")
HEADING_NUM = re.compile(r"^(#{2,3}) (\d+(?:\.\d+)?)\.")
HEADING_APP = re.compile(r"^## 附录 ([A-Z])[：:]")
SECTION_REF = re.compile(
    r"第\s*(\d+(?:\.\d+)?)((?:\s*[、，,]\s*\d+(?:\.\d+)?)*)\s*节(?:与\s*(\d+(?:\.\d+)?)\s*节)?"
)
CITE_GROUP = re.compile(r"（([^（）\n]+?)）")
YEAR = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
ALREADY_LINKED = re.compile(r"^\[(.+?)\]\(#(ref-[^)]+)\)$")


def slugify(key):
    return "ref-" + re.sub(r"[^0-9a-zA-Z]+", "-", key).strip("-").lower()


def split_frontmatter(text):
    """Return (front matter including delimiters, rest of the file)."""
    marks = list(FRONTMATTER.finditer(text))
    if not text.startswith("+++") or len(marks) < 2:
        sys.exit("expected TOML front matter delimited by +++ lines")
    end = marks[1].end()
    return text[: end + 1], text[end + 1 :]


def surname_year(s):
    """('B. Beyer, C. Jones, and N. R. Murphy, 2016') -> ('beyer', '2016')"""
    m = re.search(r"\b((?:19|20)\d{2}[a-z]?)\b", s)
    if not m:
        return None
    head = re.sub(r"\b[A-Z]\.(-[A-Z]\.)*", "", s[: m.start()].strip().rstrip(","))
    head = re.sub(r"\b(et al|and|Eds\.?)\b", " ", head)
    tokens = [t for t in re.split(r"[,\s]+", head) if t]
    return (tokens[0].lower(), m.group(1)) if tokens else None


def collect_refs(lines):
    """Anchor every reference entry; return key -> anchor plus a fuzzy index."""
    keys, fuzzy = {}, {}
    for i, line in enumerate(lines):
        if '<a id="ref-' in line:
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", line)
        if not m:
            continue
        entry = m.group(2)
        label = re.match(r"^(.+?\(\d{4}[a-z]?\))", entry)
        if not label:
            continue
        label = label.group(1)
        key = re.sub(r" \((\d{4}[a-z]?)\)$", r", \1", label)
        keys[key] = slugify(key)
        sy = surname_year(key)
        if sy and sy not in fuzzy:
            fuzzy[sy] = key
        lines[i] = f'{m.group(1)}. <a id="{slugify(key)}"></a>{entry}'
    return keys, fuzzy


def link_citations(text, keys, fuzzy, stats):
    """Link each `；`-separated citation inside one （...） group."""

    def fix(group):
        parts = []
        for part in group.group(1).split("；"):
            raw = part.strip()
            if ALREADY_LINKED.match(raw):
                stats["cites"] += 1
                parts.append(raw)
            elif raw in keys:
                parts.append(f"[{raw}](#{keys[raw]})")
                stats["cites"] += 1
            elif YEAR.search(raw) and surname_year(raw) in fuzzy:
                key = fuzzy[surname_year(raw)]
                parts.append(f"[{raw}](#{keys[key]})")
                stats["fuzzy"] += 1
            else:
                if YEAR.search(raw):
                    stats["unmatched"].append(raw)
                parts.append(raw)
        return "（" + "；".join(parts) + "）"

    return CITE_GROUP.sub(fix, text)


def link_sections(text, stats):
    def fix(m):
        nums = [m.group(1)] + re.findall(r"\d+(?:\.\d+)?", m.group(2))
        out = "第 " + "、".join(f"[{n}](#sec-{n.replace('.', '-')})" for n in nums) + " 节"
        if m.group(3):
            out += f"与 [{m.group(3)}](#sec-{m.group(3).replace('.', '-')}) 节"
        stats["sections"] += len(nums) + (1 if m.group(3) else 0)
        return out

    return SECTION_REF.sub(fix, text)


def link_simple(text, pattern, anchor, counter, stats):
    """Link `图 N` / `表 N` / `附录 X` unless it is already the text of a link."""
    def fix(m):
        stats[counter] += 1
        return f"[{m.group(0)}](#{anchor}-{m.group(1).lower()})"

    return re.sub(r"(?<!\[)" + pattern, fix, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("post")
    ap.add_argument("--check", action="store_true", help="report only, do not write")
    ap.add_argument("--verify-html", help="built page to check #fragment targets against")
    args = ap.parse_args()

    path = Path(args.post)
    head, rest = split_frontmatter(path.read_text(encoding="utf-8"))
    m = REFS_HEADING.search(rest)
    if not m:
        sys.exit("no '## 参考文献' heading found")
    body, refs = rest[: m.start()], rest[m.start() :]

    lines = refs.split("\n")
    keys, fuzzy = collect_refs(lines)
    refs = "\n".join(lines)

    stats = {"cites": 0, "fuzzy": 0, "sections": 0, "figs": 0, "tables": 0, "apps": 0, "unmatched": []}
    out = []
    for line in body.split("\n"):
        m = HEADING_NUM.match(line)
        if m and "{#sec-" not in line:
            line += " {#sec-%s}" % m.group(2).replace(".", "-")
        elif (m := HEADING_APP.match(line)) and "{#app-" not in line:
            line += " {#app-%s}" % m.group(1).lower()

        m = CAPTION_FIG.match(line)
        if m and '<a id="fig-' not in line:
            line = line.replace("> **图", '> <a id="fig-%s"></a>**图' % m.group(1), 1)
        m = CAPTION_TBL.match(line)
        if m and '<a id="tbl-' not in line:
            line = '<a id="tbl-%s"></a>' % m.group(1) + line

        # headings and captions never get in-text links
        if not (line.startswith("#") or line.startswith('> <a id="fig-') or line.startswith('<a id="tbl-')):
            line = link_sections(line, stats)
            line = link_simple(line, r"图 (\d+)", "fig", "figs", stats)
            line = link_simple(line, r"表 (\d+)", "tbl", "tables", stats)
            line = link_simple(line, r"附录 ([A-Z])", "app", "apps", stats)
            line = link_citations(line, keys, fuzzy, stats)
        out.append(line)

    new = head + "\n".join(out) + refs

    ids = set(re.findall(r'<a id="([^"]+)"></a>', new)) | set(
        re.findall(r"\{#([a-z0-9-]+)\}", new)
    )
    broken = sorted(t for t in set(re.findall(r"\]\(#([^)]+)\)", new)) if t not in ids)

    print(f"reference anchors   {refs.count('<a id=\"ref-')}")
    print(f"in-text citations   {stats['cites']} ({stats['fuzzy']} matched by surname+year fallback)")
    print(f"section refs        {stats['sections']}")
    print(f"figure/table/app refs {stats['figs']} / {stats['tables']} / {stats['apps']}")
    print(f"broken targets      {broken or 'none'}")
    if stats["unmatched"]:
        print(f"unmatched citations ({len(stats['unmatched'])}) — no reference entry found:")
        for u in sorted(set(stats["unmatched"])):
            print(f"  {u}")

    failed = bool(broken or stats["unmatched"])
    if args.verify_html:
        html = Path(args.verify_html).read_text(encoding="utf-8")
        html_ids = set(re.findall(r'id="([^"]+)"', html))
        html_broken = sorted(t for t in set(re.findall(r'href="#([^"]+)"', html)) if t not in html_ids)
        print(f"built page          {len(html_ids)} ids, broken: {html_broken or 'none'}")
        failed = failed or bool(html_broken)

    if args.check:
        sys.exit(1 if failed else 0)
    if failed:
        print("refusing to write: fix the problems above (or the conventions)", file=sys.stderr)
        sys.exit(1)
    path.write_text(new, encoding="utf-8")
    print(f"written             {path}")


if __name__ == "__main__":
    main()
