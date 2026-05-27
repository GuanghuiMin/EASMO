You are helping write the motivation section for a research paper on agent context compression.

You are given aggregated JSON statistics and representative cases from AppWorld ACON failure audits.

Write a concise motivation analysis that answers:
1. What failure modes dominate ACON failures?
2. What information does the audit model most often add back?
3. What information does the compressor most often drop again?
4. What does this imply about the real bottleneck in agent context compression?
5. Which future method directions are supported by the evidence, and which are not?

Do not invent numbers. Use only the provided aggregate statistics.
Distinguish observations from hypotheses.

Return Markdown with these headings:

# Motivation Findings
## Observation 1: ...
## Observation 2: ...
## Observation 3: ...
## Implications for Method Design
## Negative Results / What Not to Pursue
## Representative Cases

---
AGGREGATE_STATS_JSON_START
{{aggregate_stats_json}}
AGGREGATE_STATS_JSON_END

REPRESENTATIVE_CASES_JSON_START
{{representative_cases_json}}
REPRESENTATIVE_CASES_JSON_END
