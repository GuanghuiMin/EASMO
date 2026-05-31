[ ] Confirm Stage 07 passes split=train|dev into run_with_compressed_context.
[ ] If any Stage 07 behavior rows were produced before the split fix, delete and rerun them.
[ ] Confirm full_context_runs has 145 expected task attempts or document exclusions.
[ ] Confirm task inventory records why train has 89 tasks rather than official 90.
[ ] Confirm candidate generation produces:
    n_boundaries × 4 prompt_families × 9 candidates.
[ ] Confirm serial stress preserves prompt_family, not always UTCO.
[ ] Confirm transition matrix is all-task, not full-context-success-only.
[ ] Rename unconditional preserve/harm/rescue shares vs conditional rates.
[ ] Add full_context_retry for F=0 tasks if rescue is a main claim.
[ ] Save full candidate bank with split, task_id, prompt_family, candidate_id, C1 text, CK text, stress chain, pass/fail.