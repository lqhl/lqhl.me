#!/usr/bin/env python3
"""Download an arXiv HTML paper into a translation-ready working directory.

Usage:
    python3 fetch_arxiv.py <arxiv-url-or-id> <outdir>

Writes into <outdir>:
    paper.html   raw arXiv HTML (kept so re-runs are offline)
    paper.md     body as markdown, up to (not including) the bibliography,
                 with [[FIGURE: name.svg]] markers where each figure sits
    refs.md      cleaned bibliography, one numbered entry per line
    figs/*.svg   figure assets referenced from paper.md

Requires: curl, python3, beautifulsoup4, html2text.
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    import html2text
except ImportError:
    sys.exit("missing deps: python3 -m pip install --break-system-packages beautifulsoup4 html2text")

UA = "Mozilla/5.0"
ASSET_EXT = (".svg", ".png", ".jpg", ".jpeg", ".gif")


def arxiv_id(s):
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", s)
    if not m:
        sys.exit(f"cannot find an arXiv id in {s!r}")
    return m.group(1) + (m.group(2) or "")


def curl(url, dest):
    """Download with retries. Existing non-empty files are kept, so a failed
    run can simply be re-run to resume (arXiv occasionally drops TLS)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size:
        return
    for attempt in range(1, 7):
        r = subprocess.run(
            ["curl", "-sL", "-A", UA, url, "-o", str(dest)],
            check=False,
        )
        if r.returncode == 0 and dest.exists() and dest.stat().st_size:
            return
        print(f"  retry {attempt}/6: {url} (curl exit {r.returncode})", file=sys.stderr)
        time.sleep(1.5 * attempt)
    sys.exit(f"download failed: {url}\nre-run the command to resume; finished files are kept")


def clean_bibitem(item):
    """Turn a bibliography <li> into one clean citation line."""
    for div in item.find_all("div", class_=re.compile("ltx_bib_cited")):
        div.decompose()
    text = re.sub(r"\s+", " ", item.get_text()).strip()
    text = re.sub(r"\s*(Cited by|External Links):.*$", "", text).strip()
    text = text.replace(" Note: ", " ")
    links = []
    for a in item.find_all("a", href=True):
        href = a["href"]
        if href.startswith("http") and href not in links:
            links.append(href)
    if links:
        text += " [" + " ".join(f"<{u}>" for u in links[:2]) + "]"
    return text


def label_of(entry):
    """'Smith, J. (2020) Foo.' -> 'Smith, J. (2020)'."""
    m = re.match(r"^(.+?\(\d{4}[a-z]?\))", entry)
    return m.group(1) if m else entry


def normalize_label(label):
    """Edited volumes are labelled by editors, e.g.
    'B. Beyer, C. Jones, and N. R. Murphy (Eds.) (2016)'. In-text citations
    say 'Beyer et al., 2016', so collapse such labels to that form."""
    m = re.match(r"^(.+?)\s*\((\d{4}[a-z]?)\)$", label)
    if not m or ("(Eds.)" not in label and not re.match(r"^([A-Z]\.\s*)+", label)):
        return label
    names = re.sub(r"\(Eds\.\)", "", m.group(1))
    names = re.sub(r"\b[A-Z]\.(-[A-Z]\.)*", "", names)  # drop initials
    surnames = [w.strip() for w in re.split(r",|\band\b", names) if w.strip()]
    if len(surnames) > 2:
        return f"{surnames[0]} et al. ({m.group(2)})"
    if len(surnames) == 2:
        return f"{surnames[0]} and {surnames[1]} ({m.group(2)})"
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paper", help="arXiv id, abs/html/pdf URL")
    ap.add_argument("outdir")
    args = ap.parse_args()

    pid = arxiv_id(args.paper)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / "paper.html"

    if not html_path.exists():
        curl(f"https://arxiv.org/html/{pid}", html_path)

    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    article = soup.find("article")
    if article is None or not soup.find(class_=re.compile("ltx_")):
        sys.exit(
            f"{pid} has no arXiv HTML (LaTeXML) version; use the pdf-reader skill on "
            f"https://arxiv.org/pdf/{pid} instead"
        )

    for tag in soup(["nav", "footer", "script", "style", "header"]):
        tag.decompose()

    # --- figures: replace each graphic with a marker and download the asset
    figs_dir = out / "figs"
    figs_dir.mkdir(exist_ok=True)
    names = []
    for node in article.find_all(["object", "img"]):
        src = node.get("data") or node.get("src") or ""
        if not src.endswith(ASSET_EXT) or src.startswith("/static/"):
            continue
        name = Path(src).name
        if name not in names:
            curl(f"https://arxiv.org/html/{pid}/{name}", figs_dir / name)
            names.append(name)
        marker = soup.new_tag("p")
        marker.string = f"[[FIGURE: {name}]]"
        node.replace_with(marker)

    # --- bibliography: clean it up, then drop it from the body
    refs = []
    bib = article.find("section", id="bib")
    if bib is not None:
        for item in bib.find_all("li", class_=re.compile("ltx_bibitem")):
            entry = clean_bibitem(item)
            refs.append(normalize_label(label_of(entry)) + entry[len(label_of(entry)):])
        bib.decompose()

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.body_width = 0
    body = re.sub(r"\n{3,}", "\n\n", h.handle(str(article))).strip() + "\n"
    (out / "paper.md").write_text(body, encoding="utf-8")
    (out / "refs.md").write_text(
        "\n".join(f"{i}. {r}" for i, r in enumerate(refs, 1)) + "\n", encoding="utf-8"
    )

    print(f"id        {pid}")
    print(f"paper.md  {len(body)} chars, {body.count('[[FIGURE:')} figure markers")
    print(f"refs.md   {len(refs)} entries")
    print(f"figs/     {len(names)} files: {', '.join(names)}")


if __name__ == "__main__":
    main()
