from types import SimpleNamespace

from rag_utils import build_document_summary_task, merge_corpus_for_summary


def test_document_summary_removes_overlapping_chunk_text():
    first = SimpleNamespace(
        source="cv.pdf",
        page=1,
        chunk_id=0,
        text=(
            "Tahiya Zareen Education Computer Science at UAP "
            "Skills Python TypeScript Next.js GitHub RAG Projects "
            "Built an academic assistant"
        ),
    )
    second = SimpleNamespace(
        source="cv.pdf",
        page=1,
        chunk_id=1,
        text=(
            "Skills Python TypeScript Next.js GitHub RAG Projects "
            "Built an academic assistant using retrieval augmented generation."
        ),
    )

    merged = merge_corpus_for_summary([first, second], max_chars=2000)

    assert merged.count("Skills Python TypeScript") == 1
    assert "using retrieval augmented generation" in merged
    assert "Education Computer Science at UAP" in merged


def test_cv_summary_task_requests_a_profile_not_contact_header():
    source = (
        "tahiya@example.com | linkedin.com/in/example | github.com/example\n"
        "Education: Computer Science at UAP\n"
        "Technical Skills: Python, TypeScript\n"
        "Projects: UAP CSE Assistant\n"
        "Built a RAG-based university website assistant."
    )

    task = build_document_summary_task(source)

    assert "CV/resume" in task
    assert "professional/academic profile" in task
    assert "Exclude contact details" in task
    assert "do not invent seniority" in task



def test_general_summary_task_adapts_to_multiple_document_genres():
    paper = build_document_summary_task(
        "Abstract\nThis study evaluates a retrieval system.\nMethods\nWe compare two encoders.\n"
        "Results\nEncoder A improves recall.\nLimitations\nThe dataset is small."
    )
    notes = build_document_summary_task(
        "Binary search\nCompare the target with the middle element.\n"
        "If the target is smaller, continue in the left half; otherwise continue in the right half."
    )

    assert "research papers" in paper
    assert "lecture notes or textbooks" in paper
    assert "do not force a paper-style" in paper
    assert "technical documentation or procedures" in notes
    assert "do not invent facts" in notes
