# Action-verb guidance for EduGuard rewrite generation

## Purpose

Use varied, observable opening verbs appropriate to the intended cognitive operation. A verb list is a drafting aid, not a label oracle: determine Bloom alignment from the full cognitive demand and expected answer, not from the first word alone.

## Working verb bank

| Revised Bloom level | Candidate opening verbs | Generation caution |
|---|---|---|
| Remember | Identify, list, name, define, label, recall, recognize, state, match, select | Use for retrieval or recognition of learned information; avoid adding explanation or judgment. |
| Understand | Explain, summarize, describe, classify, interpret, paraphrase, illustrate, compare, outline, distinguish | The prompt should ask for meaning, interpretation, or explanation, not only recall. |
| Apply | Calculate, solve, use, demonstrate, implement, execute, compute, apply, construct, modify | Provide the necessary data/context; the student should use a method or principle in a task. |
| Analyze | Differentiate, compare, contrast, categorize, examine, investigate, separate, inspect, infer, break down | Require relationships, components, patterns, or causes to be examined. |
| Evaluate | Assess, appraise, justify, critique, judge, recommend, defend, rate, prioritize, support | State or imply a defensible criterion or evidence basis for judgment. |
| Create | Design, develop, formulate, devise, generate, compose, propose, construct, plan, produce | Ask for a new product, plan, solution, hypothesis, or synthesis—not merely a description. |

## Sources consulted

1. University of Waterloo, Centre for Teaching Excellence, “Bloom’s Taxonomy”:
   https://uwaterloo.ca/centre-for-teaching-excellence/catalogs/tip-sheets/blooms-taxonomy
2. University of Waterloo, Centre for Teaching Excellence, “Bloom’s Taxonomy Learning Activities and Assessments”:
   https://uwaterloo.ca/centre-for-teaching-excellence/catalogs/tip-sheets/blooms-taxonomy-learning-activities-and-assessments
3. University of Oklahoma, Office of Academic Assessment, “Bloom’s Taxonomy”:
   https://www.ou.edu/assessment/resources/blooms-taxonomy
4. Vanderbilt University, IRIS Center, “Digging Deeper: Learning Goals”:
   https://iris.peabody.vanderbilt.edu/module/fac/cresource/q1/p02/

## Generation and validation rules

- Every candidate rewrite should begin with a suitable action verb, not a WH-question phrase.
- Vary opening verbs within and across levels; do not create diversity by substituting a verb that changes the intended cognitive operation.
- Keep quantities, named entities, domain, scenario, constraints, and required output where they remain compatible with the target operation.
- When a target-level change necessarily alters the original task, mark that trade-off explicitly rather than claiming strict semantic equivalence.
- Flag malformed or underspecified source questions instead of silently inventing missing context.
- Store source-level, target-level, rewrite, verb, provenance, automated-review status, and human-review status separately.
- Do not treat automated screening or model-generated rewrites as human-validated gold labels.
