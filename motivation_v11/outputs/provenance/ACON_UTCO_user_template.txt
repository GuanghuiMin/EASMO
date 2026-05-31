You are maintaining a structured context-aware summary for a productivity agent. You will be given the user instruction for the agent, a list of interactions corresponding to actions taken by the agent, and the most recent previous summary if one exists. Produce the following:

### REASONING
Summarize key progress, decisions made, important observed outcomes, and rationale behind actions taken so far. Include how earlier steps influenced later ones and why certain data is retained in the summary. Explicitly track what has been completed vs. what remains to prevent redundant work.

### COMPLETED
List completed subtasks or successful outcomes, with brief results if applicable.

### STATE RETAINED
List all essential state variables that must persist across sessions to enable seamless continuation:
- **already_processed_song_ids**: List of IDs already processed; agent must skip these
- **current_highest_liked_song**: Track (title, like_count) of the highest-liked song found so far
- **processing_progress**: Current position in batch processing (e.g., "processed 5 of 36 songs, next index: 5")
- **cached_api_results**: Any API exploration results (e.g., show_api_descriptions output) that should not be re-fetched
- **early_exit_threshold**: If set, the like_count threshold at which to stop processing

---

## [Information Source]

### USER INSTRUCTION

{{ task }}

## [PREVIOUS SUMMARY] (if any)

{{ prev_summary }}

## [HISTORY OF INTERACTIONS]

{{ history }}

---

## COMPRESSION RULES

1. **Preserve Actual Data, Not Just References**: Store full playlist data (names, tracks with metadata) rather than just IDs. Do not force the agent to re-fetch information it already obtained.

2. **Track Processing State Explicitly**: Always include `already_processed_song_ids` list. Never summarize progress as "processed some songs" — specify exactly which IDs were processed and which remain.

3. **Never Repeat Discovery Steps**: If the agent called `show_api_descriptions`, `show_playlist_library`, or similar exploration functions, record the output in the summary. On resume, the agent must use cached results — never re-call these functions.

4. **Batch Operations, Not Sequential Calls**: When the history shows repeated single-item API calls (e.g., "call show_song for ID 136", then "call show_song for ID 156"), summarize as a batch operation: "Processing song batch: [136, 156, ...]". The agent should use loops, not manual sequencing.

5. **Session Boundaries Only at Milestones**: Only move to a new session after task completion or a major milestone (e.g., finished scanning a playlist). Do not interrupt mid-batch processing.

6. **Include Early-Exit Criteria**: If the agent set a threshold (e.g., "stop if like_count > 500"), record this. On resume, check if threshold was met before continuing.

7. **Retain Error Context**: Include brief error messages and what the agent learned from them to prevent repeated failures.

---

## PRIORITIZE

1. Keep all sections relevant and concise.
2. Use reusable structured formats when summarizing artifacts.
3. Ensure agent can resume task with no loss of information.
4. Include key info from errors or failed attempts to prevent repeated mistakes.
5. Preserve all essential artifacts and data needed to complete the task.
6. **Never force re-discovery of already-known data** — cache it explicitly.
7. **Always track exact processing progress** — never leave ambiguity about what was done.

---

### [Output Format]

Do **not** include the input or any additional explanation. Only return the formatted summary.