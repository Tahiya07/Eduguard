# Pass-3 repair rules (applied to every eligible example)

These rules are written for whoever reviews pass 3, including the reviewer applying them. They restate the brief so it doesn't have to be re-read.

**Goal.** Keep what the question is about and change what the learner must do. A model should not be able to tell the level from a memorised verb or template.

## Content rules

- **No new facts.** Use only content already in the source question or in the row being rewritten. Do not add numbers, data, entities, criteria, diagrams or scenarios.
  - Authored-scenario and authored-object rows keep their existing values exactly.
  - Generic sources stay generic.
- **Deliberately false Evaluate claims stay.** Keep the stated wrong value, claim or object exactly; only the framing around it may change. The claim must remain something to judge.
- **Label-only rows** that cite "supplied/specified" material keep that reference; nothing absent may be invented.

## What each level must ask the learner to do

| Level | The learner must … |
|---|---|
| Remember | retrieve or recognise a fact, term, rule or component. The answer must not appear in the question. |
| Understand | explain, interpret, summarise, classify, exemplify, compare or infer at the level of meaning. No new-case solving and no judgement. |
| Apply | carry out a procedure or use a concept in a specific case or set of conditions. |
| Analyze | break something into parts and work out relationships, causes, structure, evidence or assumptions, or locate an error. Explaining a mechanism is not enough. |
| Evaluate | reach a judgement against stated or defensible criteria and justify it: verify, critique, recommend, rank or weigh. |
| Create | produce a new artefact, plan, model, hypothesis or solution. Reproducing a standard algorithm is not enough. |

## Question form and verbs (pass 3.1, user directive 2026-11-25; replaces the pass-3 OBE verb-bank rule)

- **No per-level verb bank.** A row no longer has to begin with a verb from its level's bank. `obe_verb_bank.json` is kept only as a reference list and for diagnostics.
- **Any verb at any level, if the cognitive demand justifies it.** Examples: "Trace" for executing a procedure (Apply) or following a cause (Analyze); "Compare" for comparing meanings (Understand) or comparing factors to attribute an effect (Analyze); "Check" for judging a result against criteria (Evaluate); "Find" or "Modify" for producing something new (Create).
- **Imperative and interrogative forms are both allowed**, including wh-questions ("Why does …?", "What is the … when …?", "How would you design …?") and judgement questions ("Is the reported value correct? Explain your verdict."). Scenario sentences may come first.
- **The level is set by the cognitive operation the learner must perform** (revised Bloom: recognize/recall; interpret, exemplify, classify, summarize, infer, compare, explain; execute, implement; differentiate, organize, attribute; check, critique; generate, plan, produce), never by the opening word.
- **Do not reword to move classifier scores.** Wording changes are made only to state the task naturally and keep the cognitive distinction between levels clear. Predictability numbers are reported as diagnostics.
- **No new templates.** Vary the tails of judgement questions ("Justify your answer", "Explain your verdict", "Show the check …"). Do not let one form, such as the question mark, become a level marker.
- The automated format check only rejects empty, one-word or visibly truncated rows. Bloom alignment is reviewed, not machine-checked.

## Sibling checks

- The five transformations of one example must differ in cognitive operation, not only in wording.
- **Understand vs Analyze:** the Analyze row must require decomposition, comparison of factors, causal attribution or error location that the Understand row does not.

## Process

- **Change only what is needed.** Leave a row unchanged if it is aligned and its wording doesn't add to template dominance.
- **Targets are ceilings, not quotas:**

  | Level | Ceiling on the dominant verb(s) |
  |---|---|
  | Understand | Explain ≤ about 50% |
  | Analyze | Examine ≤ about 40% |
  | Evaluate | Verify and Assess ≤ about 30% each |
  | Create | Design + Develop ≤ about 55% |
  | Remember | Define / State / Identify ≤ about 75% |

- **Exclude rather than invent:** if no valid row can be written from the available content, exclude it and give the reason.
- **Patch file format:**

  ```
  @@ <id>|<Level>|<category>|<short reason>
  <full new text>
  ```

  To relabel a source, use `@@ <id>|SOURCE_LABEL|<new level>|<reason>`. Rows the source relabel makes invalid are handled with `@@ <id>|<Level>|DROP|<reason>`, and new rows with the normal format.
- **Review record:** each patch file starts with `#reviewed <comma-separated ids>`, listing every example reviewed in that batch, including those with no changes.
