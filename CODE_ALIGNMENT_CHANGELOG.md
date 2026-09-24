# R²-Mem code alignment notes

Date: 2026-09-24  
Base snapshot: public repository commit `e0fe67c7fd8549587a0e00566f02954eb618f26c`

This update makes small changes to the released main framework so that the implementation better matches the ICLR manuscript. It does not turn the repository into an automated pipeline for regenerating every paper table.

## A01 — Shared prompts for GAM and R²-Mem

**Files**

- `gam/agents/research_agent.py`

**Change**

GAM now imports the same R²-Mem Planning, Integration, InfoCheck, and follow-up-query templates used by the experience-aware agent. For GAM, the experience field explicitly states that no prior experience is provided. Thus, the prompt structure is shared and the intended difference is whether retrieved experience is present.

## A02 — Aligned Reflection failure behavior

**Files**

- `gam/agents/research_agent.py`

**Change**

When GAM encounters an exception in Reflection, it now returns `enough=True`, matching the existing R²-Mem behavior and stopping the current search instead of continuing with a different control flow.

## A03 — Fixed return-value unpacking

**Files**

- `exp/self_reflection.py`

**Meaning of the issue**

`self_reflection(...)` normally returns three values: generated experiences, total tokens, and Learner/self-reflection tokens. The NarrativeQA and HotpotQA branches previously received only two values, which raises `ValueError: too many values to unpack`. Because the surrounding code catches broad exceptions, the failure could be hidden and the experience bank could remain empty.

**Change**

All three dataset branches now receive the same three values. Early exits from `self_reflection(...)` also return the same three-part structure.

## A04 — NarrativeQA split aligned with the paper

**Files**

- `exp/self_reflection.py`
- `exp/eval/exp_scripts/exp_narrativeqa.sh`

**Change**

For the 300-question sampled set, the experience-construction portion is now the first 60 questions (20%), and R²-Mem evaluation begins at index 60. This replaces the previous 30-question (10%) split.

## A05 — Single-seed manual runs

**Files**

- `exp/self_reflection.py`
- `exp/eval/locomo.py`
- `scripts/eval_narrativeqa.sh`
- `exp/eval/exp_scripts/exp_narrativeqa.sh`

**Change**

Each invocation uses one active seed, `43`. Seeds `42` and `44` are listed as explicit alternatives for separate manual runs; no automatic multi-seed loop was added.

- LoCoMo selects one source conversation with the active seed and evaluates the remaining conversations.
- HotpotQA uses seeded stratified sampling and saves the selected IDs as before.
- NarrativeQA baseline and R²-Mem scripts use the same active seed.
- Python, NumPy, and PyTorch sampling are seeded in the experience-construction script.

With seed `43`, the LoCoMo source remains Conv-26, preserving the representative configuration already used by the released code.

## A06 — Temperatures aligned with the paper

**Files**

- `exp/self_reflection.py`
- `exp/eval/locomo.py`

**Change**

The Learner temperature was changed from `0.5` to `0.3`, and the LoCoMo online R²-Mem generator temperature was changed from `0.2` to `0.3`. These now match the manuscript. The GPT-4o Evaluator remains at `0.2`.

## A08 — Self-Evo clarification

**Files**

- `exp/self_reflection.py`
- `exp/judge_model.py`

**Change**

Comments now state that Self-Evo uses the same backbone as the trajectory Evaluator. Reference answers are restricted to the offline experience-construction split. The backbone evaluates those construction trajectories, forms experience through the same reflection pipeline, and later uses the fixed experience bank without evaluation answers or parameter updates.

## Deliberately unchanged in this pass

- A07: metric handling for failed samples.
- A09: environment, paths, dependencies, and one-command execution.
- A10: automatic table generation and result provenance.

The repository therefore remains a released main-framework snapshot rather than a complete one-command reproduction package.

## Validation

- Python syntax compilation passed for `gam/`, `exp/`, and `eval/` with an isolated bytecode cache.
- Both edited shell scripts passed `bash -n`.
- All calls to `self_reflection(...)` now use three-value unpacking.
- The shared R²-Mem prompt templates format successfully with GAM's explicit no-experience value.
