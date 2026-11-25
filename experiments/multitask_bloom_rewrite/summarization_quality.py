"""Quality rules for the arXiv summarization task (ccdv/arxiv-summarization).

Background (verified in the 2026-11-25 audit):
* The rows come from ccdv/arxiv-summarization, which redistributes the Cohan et al. (2018) arXiv corpus. Its
  conversion replaced every LaTeX math span with an ``@xmathN`` token and every citation with ``@xcite``.
  The math text itself was never kept, in articles or abstracts.
* Abstract placeholders are numbered independently of the article (most abstracts restart at ``@xmath0``), so an
  abstract placeholder cannot be mapped to an article placeholder, and neither carries an expression.
* Rows carry no arXiv identifier (``id`` is a stream position and ``title`` is empty), so they cannot be traced
  to the original LaTeX sources.

Consequently no placeholder is recoverable from available data. ``apply_verified_recoveries`` exists so that a
*verified* external mapping (exact example ID, split, field SHA-256 and token) can be applied later. It never
guesses, and it rejects any entry that does not match the row exactly.

Eligibility policy (``POLICY_VERSION``). The abstract is the training *target*, so it must be clean. The article
is the *input*: its placeholders are kept (they are pervasive and the model is never trained to emit them),
unless the input is materially corrupted. The thresholds were fixed from the train split only; validation and
test are filtered by the same rules and are never filtered on model performance.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

POLICY_VERSION = "arxiv_summarization_quality_v1_20261125"
SOURCE_DATASET = "ccdv/arxiv-summarization"

MATH_PLACEHOLDER = re.compile(r"@xmath\d+")
CONVERSION_PLACEHOLDER = re.compile(r"@[a-z]+")          # @xcite, @noop, @headings, @url, ... (and @xmath)
LATEX_OR_MARKUP_RESIDUE = re.compile(r"\\[a-zA-Z]+|\[\s*firstpage\s*\]|width\s*=|\\\]|\$|#\s*\d")
METADATA_APPENDED = re.compile(r"\bpacs\b|\bkeywords?\s*:|\be-?mail\b|subject headings|\bmsc\b", re.I)
BIBLIOGRAPHY_REFERENCE = re.compile(r"\[\s*\d+(\s*[,-]\s*\d+)*\s*\]")
COMPLETE_ENDING = re.compile(r"[.?!][\"')\]]?\s*$")

MIN_ABSTRACT_WORDS = 15
MIN_ARTICLE_WORDS = 300
MAX_ARTICLE_MATH_PLACEHOLDER_DENSITY = 0.10   # train p90 = 0.074; above 10% of tokens the input text is mostly missing math

RULES: tuple[tuple[str, str], ...] = (
    ("abstract_math_placeholder", "Target abstract contains @xmath placeholders for math that was removed upstream and cannot be recovered."),
    ("abstract_other_conversion_placeholder", "Target abstract contains other LaTeX-conversion placeholders (@xcite, @noop, @url, ...)."),
    ("abstract_latex_or_markup_residue", "Target abstract contains LaTeX commands, figure-caption fragments, [ firstpage ] or macro residue."),
    ("abstract_metadata_appended", "Target abstract has PACS numbers, keywords, e-mail addresses or subject headings appended."),
    ("abstract_bibliography_reference", "Target abstract cites numbered references ([ n ]) from a bibliography that is not in the row."),
    ("abstract_incomplete_ending", "Target abstract does not end with a complete sentence (appended keyword/affiliation lists or truncation)."),
    ("abstract_too_short", f"Target abstract has fewer than {MIN_ABSTRACT_WORDS} words."),
    ("article_not_paper_body", f"Input article has fewer than {MIN_ARTICLE_WORDS} words (acknowledgements, references or a fragment, not the paper body)."),
    ("article_math_placeholder_density", f"More than {MAX_ARTICLE_MATH_PLACEHOLDER_DENSITY:.0%} of input article tokens are unrecoverable @xmath placeholders."),
    ("abstract_copied_in_article", "The target abstract appears verbatim in the input article (copy shortcut)."),
)
RULE_DESCRIPTIONS = dict(RULES)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_sha256(row: dict[str, Any]) -> str:
    """Canonical row fingerprint (key order independent)."""
    return sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))


def math_placeholders(text: str) -> list[str]:
    return MATH_PLACEHOLDER.findall(text or "")


def article_placeholder_density(article: str) -> float:
    words = (article or "").split()
    return len(math_placeholders(article)) / max(1, len(words))


def sft_target(row: dict[str, Any]) -> str:
    """The supervised assistant text (everything after the masked prompt)."""
    sft, prompt = row.get("sft_text", ""), row.get("prompt_text", "")
    if not sft.startswith(prompt):
        raise ValueError(f"{row.get('id')}: prompt_text is not a prefix of sft_text")
    return sft[len(prompt):]


def detect_issues(row: dict[str, Any]) -> list[str]:
    """Return the rule names a summarization row violates (empty list = eligible)."""
    abstract = str(row.get("abstract") or "").strip()
    article = str(row.get("article") or "")
    issues = []
    if math_placeholders(abstract):
        issues.append("abstract_math_placeholder")
    if CONVERSION_PLACEHOLDER.search(MATH_PLACEHOLDER.sub(" ", abstract)):
        issues.append("abstract_other_conversion_placeholder")
    if LATEX_OR_MARKUP_RESIDUE.search(abstract):
        issues.append("abstract_latex_or_markup_residue")
    if METADATA_APPENDED.search(abstract):
        issues.append("abstract_metadata_appended")
    if BIBLIOGRAPHY_REFERENCE.search(abstract):
        issues.append("abstract_bibliography_reference")
    if not COMPLETE_ENDING.search(abstract):
        issues.append("abstract_incomplete_ending")
    if len(abstract.split()) < MIN_ABSTRACT_WORDS:
        issues.append("abstract_too_short")
    if len(article.split()) < MIN_ARTICLE_WORDS:
        issues.append("article_not_paper_body")
    if article_placeholder_density(article) > MAX_ARTICLE_MATH_PLACEHOLDER_DENSITY:
        issues.append("article_math_placeholder_density")
    if len(abstract) > 80 and abstract[:120] in article:
        issues.append("abstract_copied_in_article")
    # Defence in depth: the supervised text must never contain a math placeholder.
    if "sft_text" in row and math_placeholders(sft_target(row)) and "abstract_math_placeholder" not in issues:
        issues.append("abstract_math_placeholder")
    return issues


class RecoveryError(ValueError):
    """A recovery entry does not match its row exactly."""


def apply_verified_recoveries(row: dict[str, Any], entries: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replace placeholders with verified expressions; return (new_row, applied_log).

    Each entry must give ``id``, ``split``, ``field`` ('abstract' or 'article'), ``field_sha256`` (SHA-256 of the
    field's original text), ``placeholder`` (e.g. '@xmath3'), ``expression`` and ``provenance``. Entries for other
    rows are ignored. An entry for this row is applied only if its field hash matches and its placeholder occurs
    as a whole token; otherwise RecoveryError is raised. Nothing is ever inferred.
    """
    mine = [e for e in entries if e.get("id") == row.get("id") and e.get("split") == row.get("split")]
    if not mine:
        return row, []
    new = dict(row)
    applied = []
    for e in mine:
        field = e.get("field")
        if field not in ("abstract", "article"):
            raise RecoveryError(f"{row.get('id')}: unsupported field {field!r}")
        for key in ("placeholder", "expression", "provenance", "field_sha256"):
            if not e.get(key):
                raise RecoveryError(f"{row.get('id')}: recovery entry missing {key!r}")
        if sha256_text(row[field]) != e["field_sha256"]:
            raise RecoveryError(f"{row.get('id')}: {field} SHA-256 does not match the recovery entry; refusing to apply")
        token = re.compile(re.escape(e["placeholder"]) + r"(?!\d)")
        if not MATH_PLACEHOLDER.fullmatch(e["placeholder"]) or not token.search(new[field]):
            raise RecoveryError(f"{row.get('id')}: placeholder {e['placeholder']!r} not found as a whole token in {field}")
        new[field] = token.sub(lambda _m: e["expression"], new[field])
        applied.append({"field": field, "placeholder": e["placeholder"], "provenance": e["provenance"]})
    return new, applied


def recovery_status(row: dict[str, Any], applied: list[dict[str, Any]]) -> str:
    if applied and not math_placeholders(row.get("abstract", "")):
        return "recovered_verified"
    if math_placeholders(row.get("abstract", "")):
        return "unrecoverable_no_source_expression"
    return "not_needed"
