"""The QA path must keep grounded answers that reuse the notes' wording (regression: they became refusals)."""
from rag_utils import _REFUSAL, sanitize_rag_answer

NOTES = [
    "Deadlock happens when two or more processes each hold a resource and wait for a resource held by "
    "another, so none can continue. The four Coffman conditions are mutual exclusion, hold and wait, "
    "no preemption, and circular wait."
]
ANSWER = (
    "Deadlock occurs when two or more processes each hold a resource and wait for a resource held by another, "
    "so none can continue. Coffman's four conditions are mutual exclusion, hold and wait, no preemption, and circular wait."
)


def test_qa_keeps_faithful_answer_that_reuses_note_wording():
    assert sanitize_rag_answer(ANSWER, NOTES, on_copy="trim") == ANSWER


def test_summary_mode_still_refuses_verbatim_dump_so_extractive_fallback_runs():
    assert sanitize_rag_answer(ANSWER, NOTES) == _REFUSAL


def test_qa_trims_long_verbatim_dump_instead_of_refusing():
    long_notes = [" ".join(f"Sentence {i} explains part {i} of the topic in plain words." for i in range(12))]
    out = sanitize_rag_answer(long_notes[0], long_notes, on_copy="trim")
    assert out != _REFUSAL
    assert out.count(".") == 3


def test_context_labels_are_still_stripped():
    out = sanitize_rag_answer("[1] Normalization reduces redundancy.", ["unrelated"], on_copy="trim")
    assert out == "Normalization reduces redundancy."


def test_empty_answer_is_still_a_refusal():
    assert sanitize_rag_answer("  ", NOTES, on_copy="trim") == _REFUSAL


# --- notes-only answering -------------------------------------------------------------
from rag_utils import keep_supported_sentences  # noqa: E402

DB_NOTES = ["Database normalization organizes tables to reduce redundancy and update anomalies."]


def test_outside_knowledge_answer_becomes_refusal():
    assert keep_supported_sentences("The capital of France is Paris.", NOTES) == _REFUSAL


def test_supported_answer_is_kept():
    assert keep_supported_sentences(ANSWER, NOTES) == ANSWER


def test_unsupported_sentence_is_dropped_supported_one_kept():
    answer = ("Normalization organizes tables to reduce redundancy and update anomalies. "
              "It was invented by Edgar Codd at IBM in 1970 for mainframe payroll systems.")
    assert keep_supported_sentences(answer, DB_NOTES) == (
        "Normalization organizes tables to reduce redundancy and update anomalies."
    )


def test_refusal_passes_through():
    assert keep_supported_sentences(_REFUSAL, NOTES) == _REFUSAL
