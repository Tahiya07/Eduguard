import json
import re
import time
from pathlib import Path

from llama_cpp import Llama


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = Path(
    r"C:\Users\tahiy\PycharmProjects\Eduguard\models\qwen.gguf"
)

FINETUNED_MODEL = Path(
    r"C:\Users\tahiy\PycharmProjects\Eduguard\models\qwen15b_multitask_v3_q4_k_m.gguf"
)

OUTPUT_FILE = Path(
    r"C:\Users\tahiy\PycharmProjects\Eduguard\rewrite_outputs\rewrite_model_comparison.jsonl"
)

N_CTX = 2048
MAX_TOKENS = 120

TEMPERATURE = 0.2
TOP_P = 0.9

# Number of questions to test
MAX_SAMPLES = 100


# ============================================================
# TEST QUESTIONS
# Replace/add your own real questions here
# ============================================================

TEST_QUESTIONS = [
    {
        "id": "q001",
        "question": "Explain about ROI",
        "target_level": "Analyze",
    },
    {
        "id": "q002",
        "question": "Explain the importance of database normalization.",
        "target_level": "Analyze",
    },
    {
        "id": "q003",
        "question": "Describe how a compiler works.",
        "target_level": "Analyze",
    },
    {
        "id": "q004",
        "question": "Explain the concept of blockchain.",
        "target_level": "Analyze",
    },
    {
        "id": "q005",
        "question": "Explain how TCP works.",
        "target_level": "Analyze",
    },
    {
        "id": "q006",
        "question": "Describe the role of an operating system.",
        "target_level": "Analyze",
    },
    {
        "id": "q007",
        "question": "Explain photosynthesis.",
        "target_level": "Analyze",
    },
    {
        "id": "q008",
        "question": "Explain the causes of inflation.",
        "target_level": "Analyze",
    },
    {
        "id": "q009",
        "question": "Explain the working principle of a CPU.",
        "target_level": "Analyze",
    },
    {
        "id": "q010",
        "question": "Describe object-oriented programming.",
        "target_level": "Analyze",
    },
]


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """You are an academic question rewriting assistant.

Rewrite the given question for the requested Bloom's Taxonomy cognitive level.

STRICT RULES:

1. Preserve the original topic and meaning.
2. Do not introduce facts that are not present in the original.
3. Do not invent components, causes, effects, relationships, or processes.
4. Do not use generic templates mechanically.
5. Change the cognitive operation, not the underlying subject.
6. Keep the result concise and natural.
7. Return ONLY the rewritten question.
"""


def build_prompt(question, target_level):
    return f"""Original question:
{question}

Target Bloom level:
{target_level}

Rewrite the question at the target Bloom level while preserving its meaning.

Rewritten question:"""


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(path, name):
    print(f"\nLoading {name}:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    model = Llama(
        model_path=str(path),
        n_ctx=N_CTX,
        n_threads=8,
        n_batch=256,
        verbose=False,
    )

    return model


# ============================================================
# GENERATION
# ============================================================

def generate(model, question, target_level):
    prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + build_prompt(question, target_level)
    )

    start = time.perf_counter()

    output = model(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=[
            "\n\n",
            "Original question:",
            "Target Bloom level:",
        ],
    )

    elapsed = time.perf_counter() - start

    text = output["choices"][0]["text"].strip()

    # Remove accidental labels
    text = re.sub(
        r"^(Rewritten question:|Answer:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    return text, elapsed


# ============================================================
# SIMPLE FAILURE DETECTION
# ============================================================

GENERIC_PATTERNS = [
    r"parts of .+ interact",
    r"parts of .+ work together",
    r"produce the overall effect",
    r"overall effect",
    r"interact to",
    r"components of .+ interact",
    r"analyze how the parts",
    r"analyze how .+ interact",
    r"examine how the parts",
]


def detect_generic_template(text):
    text_lower = text.lower()

    matches = []

    for pattern in GENERIC_PATTERNS:
        if re.search(pattern, text_lower):
            matches.append(pattern)

    return matches


def extract_keywords(question):
    """
    Very lightweight topic preservation check.

    Removes common instructional words and keeps
    meaningful content words.
    """

    stopwords = {
        "explain",
        "describe",
        "discuss",
        "about",
        "the",
        "a",
        "an",
        "how",
        "what",
        "why",
        "is",
        "are",
        "of",
        "to",
        "and",
        "in",
        "on",
        "for",
        "with",
        "does",
        "do",
    }

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9-]*\b",
        question.lower(),
    )

    return [
        word
        for word in words
        if word not in stopwords and len(word) > 2
    ]


def topic_overlap(original, rewritten):
    original_keywords = extract_keywords(original)
    rewritten_lower = rewritten.lower()

    if not original_keywords:
        return 1.0

    matched = sum(
        1
        for word in original_keywords
        if word in rewritten_lower
    )

    return matched / len(original_keywords)


def classify_failure(original, rewritten):
    failures = []

    generic_matches = detect_generic_template(rewritten)

    if generic_matches:
        failures.append("GENERIC_TEMPLATE")

    overlap = topic_overlap(original, rewritten)

    if overlap < 0.5:
        failures.append("LOW_TOPIC_OVERLAP")

    if len(rewritten.split()) > max(35, len(original.split()) * 5):
        failures.append("OVER_GENERATION")

    if not rewritten:
        failures.append("EMPTY_OUTPUT")

    return failures


# ============================================================
# RUN COMPARISON
# ============================================================

def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("EDUGUARD REWRITING MODEL COMPARISON")
    print("=" * 70)

    base_model = load_model(
        BASE_MODEL,
        "BASE MODEL",
    )

    finetuned_model = load_model(
        FINETUNED_MODEL,
        "FINE-TUNED MODEL",
    )

    samples = TEST_QUESTIONS[:MAX_SAMPLES]

    results = []

    for i, item in enumerate(samples, start=1):

        question = item["question"]
        target = item["target_level"]

        print("\n" + "-" * 70)
        print(f"[{i}/{len(samples)}]")
        print(f"Original : {question}")
        print(f"Target   : {target}")

        # ----------------------------------------------------
        # BASE
        # ----------------------------------------------------

        base_output, base_time = generate(
            base_model,
            question,
            target,
        )

        # ----------------------------------------------------
        # FINE-TUNED
        # ----------------------------------------------------

        ft_output, ft_time = generate(
            finetuned_model,
            question,
            target,
        )

        base_failures = classify_failure(
            question,
            base_output,
        )

        ft_failures = classify_failure(
            question,
            ft_output,
        )

        base_overlap = topic_overlap(
            question,
            base_output,
        )

        ft_overlap = topic_overlap(
            question,
            ft_output,
        )

        result = {
            "id": item["id"],
            "original": question,
            "target_level": target,

            "base_model": {
                "output": base_output,
                "latency_seconds": round(base_time, 4),
                "topic_overlap": round(base_overlap, 4),
                "failures": base_failures,
            },

            "finetuned_model": {
                "output": ft_output,
                "latency_seconds": round(ft_time, 4),
                "topic_overlap": round(ft_overlap, 4),
                "failures": ft_failures,
            },
        }

        results.append(result)

        print("\nBASE:")
        print(base_output)

        print("\nFINE-TUNED:")
        print(ft_output)

        if ft_failures:
            print("\n⚠ Fine-tuned failures:")
            for failure in ft_failures:
                print(f"  - {failure}")

        print(
            f"\nTopic overlap:"
            f" base={base_overlap:.2f}"
            f" | fine-tuned={ft_overlap:.2f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    base_template_failures = sum(
        "GENERIC_TEMPLATE"
        in r["base_model"]["failures"]
        for r in results
    )

    ft_template_failures = sum(
        "GENERIC_TEMPLATE"
        in r["finetuned_model"]["failures"]
        for r in results
    )

    base_topic_failures = sum(
        "LOW_TOPIC_OVERLAP"
        in r["base_model"]["failures"]
        for r in results
    )

    ft_topic_failures = sum(
        "LOW_TOPIC_OVERLAP"
        in r["finetuned_model"]["failures"]
        for r in results
    )

    base_avg_overlap = (
        sum(
            r["base_model"]["topic_overlap"]
            for r in results
        )
        / len(results)
    )

    ft_avg_overlap = (
        sum(
            r["finetuned_model"]["topic_overlap"]
            for r in results
        )
        / len(results)
    )

    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nSamples: {len(results)}")

    print("\nGeneric-template failures:")
    print(f"  Base model      : {base_template_failures}")
    print(f"  Fine-tuned      : {ft_template_failures}")

    print("\nLow-topic-overlap failures:")
    print(f"  Base model      : {base_topic_failures}")
    print(f"  Fine-tuned      : {ft_topic_failures}")

    print("\nAverage topic overlap:")
    print(f"  Base model      : {base_avg_overlap:.3f}")
    print(f"  Fine-tuned      : {ft_avg_overlap:.3f}")

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("\nDone.")


if __name__ == "__main__":
    main()