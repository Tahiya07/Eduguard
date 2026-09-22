#!/usr/bin/env python3
"""Render review PDFs from manuscript .tex for quick reading (not final typesetting)."""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def latex_to_text(s: str) -> str:
    s = s.replace("~", " ")
    s = s.replace("\\&", "&")
    s = s.replace("\\%", "%")
    s = s.replace("\\_", "_")
    s = s.replace("\\#", "#")
    s = re.sub(r"\\textbf\{([^}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"<i>\1</i>", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"<i>\1</i>", s)
    s = re.sub(r"\\texttt\{([^}]*)\}", r"<font face='Courier'>\1</font>", s)
    s = re.sub(r"\$([^$]+)\$", r"\1", s)
    s = s.replace("{=}", "=")
    s = s.replace("\\,", ",")
    s = re.sub(r"\\cite\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"[Fig/Table]", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", "", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_env(tex: str, env: str) -> str | None:
    m = re.search(rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}", tex, re.S)
    return m.group(1).strip() if m else None


def parse_title(tex: str) -> str:
    m = re.search(r"\\title\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", tex)
    if not m:
        m = re.search(r"\\title\{(.+?)\}\s*\n", tex, re.S)
    return latex_to_text(m.group(1)) if m else "Untitled"


def parse_sections(tex: str) -> list[tuple[str, str]]:
    # strip preamble / frontmatter noise for elsevier
    body = tex
    if "\\begin{document}" in body:
        body = body.split("\\begin{document}", 1)[1]
    if "\\end{document}" in body:
        body = body.split("\\end{document}", 1)[0]
    # remove figure/table environments (handled separately for images)
    body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "\n[FIGURE]\n", body, flags=re.S)
    body = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", "\n[TABLE — see manuscript.tex / tables/*.tex]\n", body, flags=re.S)
    body = re.sub(r"\\begin\{itemize\}.*?\\end\{itemize\}", lambda m: _itemize(m.group(0)), body, flags=re.S)
    body = re.sub(r"\\maketitle", "", body)
    body = re.sub(r"\\linenumbers", "", body)
    body = re.sub(r"\\bibliographystyle\{[^}]*\}", "", body)
    body = re.sub(r"\\bibliography\{[^}]*\}", "", body)
    body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}", "", body)

    parts: list[tuple[str, str]] = []
    abs_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if abs_m:
        parts.append(("Abstract", latex_to_text(abs_m.group(1))))
        body = body[: abs_m.start()] + body[abs_m.end() :]

    kw = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", body, re.S)
    if kw:
        parts.append(("Index Terms", latex_to_text(kw.group(1))))
        body = body[: kw.start()] + body[kw.end() :]
    kw2 = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", body, re.S)
    if kw2:
        parts.append(("Keywords", latex_to_text(kw2.group(1).replace("\\sep", ";"))))
        body = body[: kw2.start()] + body[kw2.end() :]

    # drop author / frontmatter leftovers
    body = re.sub(r"\\begin\{frontmatter\}.*?\\end\{frontmatter\}", "", body, flags=re.S)
    body = re.sub(r"\\author\{.*?\}", "", body, flags=re.S)
    body = re.sub(r"\\IEEEauthorblock[NA]\{.*?\}", "", body, flags=re.S)
    body = re.sub(r"\\affiliation\[[^\]]*\]\{.*?\}", "", body, flags=re.S)
    body = re.sub(r"\\journal\{.*?\}", "", body, flags=re.S)
    body = re.sub(r"\\title\{.*?\}", "", body, flags=re.S)

    chunks = re.split(r"\\(?:section|subsection)\*?\{([^}]*)\}", body)
    # chunks[0] preamble junk, then (title, content) pairs
    i = 1
    while i + 1 < len(chunks):
        title = latex_to_text(chunks[i])
        content = chunks[i + 1]
        # clean comments
        content = re.sub(r"(?m)^%.*$", "", content)
        paras = []
        for block in re.split(r"\n\s*\n", content):
            t = latex_to_text(block)
            if t and t not in {"[FIGURE]", "[TABLE — see manuscript.tex / tables/*.tex]"}:
                if t.startswith("[FIGURE]") or t.startswith("[TABLE"):
                    paras.append(t)
                elif len(t) > 2:
                    paras.append(t)
            elif t.startswith("[FIGURE]") or t.startswith("[TABLE"):
                paras.append(t)
        text = "\n\n".join(paras).strip()
        if title:
            parts.append((title, text))
        i += 2
    return parts


def _itemize(block: str) -> str:
    items = re.findall(r"\\item\s+(.*?)(?=\\item|\\end\{itemize\})", block, re.S)
    lines = []
    for it in items:
        lines.append("• " + latex_to_text(it))
    return "\n\n".join(lines)


def figure_paths(fig_dir: Path) -> list[Path]:
    preferred = [
        "fig01_system_architecture.png",
        "fig02_bloom_confusion_matrix.png",
        "fig03_fedavg_fedprox_learning_curves.png",
        "fig04_source_target_accuracy.png",
        "fig05_source_target_fully_validated.png",
        "fig06_multitask_task_performance.png",
        "fig07_gguf_deployment.png",
        "fig08_privacy_ablation.png",
    ]
    out = []
    for name in preferred:
        p = fig_dir / name
        if p.is_file():
            out.append(p)
    return out


def build_pdf(tex_path: Path, out_pdf: Path, venue: str) -> None:
    tex = tex_path.read_text(encoding="utf-8")
    title = parse_title(tex)
    sections = parse_sections(tex)
    fig_dir = tex_path.parent / "figures"

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Venue",
            parent=styles["Normal"],
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1",
            parent=styles["Heading1"],
            fontSize=12,
            leading=15,
            spaceBefore=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJust",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Note",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#666666"),
            spaceAfter=10,
        )
    )

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=title,
        author="EduGuard paper pack (review rendering)",
    )
    story = []
    story.append(Paragraph(title, styles["TitleCenter"]))
    story.append(
        Paragraph(
            f"<b>{venue}</b> — review PDF rendered from LaTeX source for reading.<br/>"
            "Not the final Overleaf/IEEEtran/elsarticle typesetting. "
            "Figures appended at the end.",
            styles["Venue"],
        )
    )
    story.append(
        Paragraph(
            f"Source: <font face='Courier'>{tex_path.as_posix()}</font>",
            styles["Note"],
        )
    )

    for heading, body in sections:
        story.append(Paragraph(heading, styles["H1"]))
        if not body:
            continue
        for para in body.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if para.startswith("[FIGURE]") or para.startswith("[TABLE"):
                story.append(Paragraph(f"<i>{para}</i>", styles["Note"]))
            else:
                # escape residual < > except our tags
                safe = para.replace("<b>", "«b»").replace("</b>", "«/b»")
                safe = safe.replace("<i>", "«i»").replace("</i>", "«/i»")
                safe = safe.replace("<font face='Courier'>", "«f»").replace("</font>", "«/f»")
                safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                safe = (
                    safe.replace("«b»", "<b>")
                    .replace("«/b»", "</b>")
                    .replace("«i»", "<i>")
                    .replace("«/i»", "</i>")
                    .replace("«f»", "<font face='Courier'>")
                    .replace("«/f»", "</font>")
                )
                story.append(Paragraph(safe, styles["BodyJust"]))

    figs = figure_paths(fig_dir)
    if figs:
        story.append(PageBreak())
        story.append(Paragraph("Appendix: Paper Figures", styles["H1"]))
        story.append(
            Paragraph(
                "Images generated/reused by scripts/final_paper_experiments.py.",
                styles["Note"],
            )
        )
        for fp in figs:
            story.append(Paragraph(fp.name, styles["Note"]))
            try:
                img = Image(str(fp), width=6.2 * inch, height=3.6 * inch, kind="proportional")
                # constrain max height
                max_w, max_h = 6.2 * inch, 4.2 * inch
                iw, ih = img.imageWidth, img.imageHeight
                scale = min(max_w / iw, max_h / ih)
                img.drawWidth = iw * scale
                img.drawHeight = ih * scale
                story.append(img)
            except Exception as e:
                story.append(Paragraph(f"(Could not embed {fp.name}: {e})", styles["Note"]))
            story.append(Spacer(1, 12))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    print(f"Wrote {out_pdf}")


def main() -> None:
    build_pdf(
        PAPER / "ieee_openjournal" / "manuscript.tex",
        PAPER / "ieee_openjournal" / "manuscript_review.pdf",
        "IEEE Open Journal style draft",
    )
    build_pdf(
        PAPER / "elsevier" / "manuscript.tex",
        PAPER / "elsevier" / "manuscript_review.pdf",
        "Elsevier elsarticle-style draft",
    )


if __name__ == "__main__":
    main()
