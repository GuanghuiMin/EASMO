# Micro-diff case study — `287e338_2` / `ACON_UT` / round `CK`

## Headline numbers

| metric | value |
|---|---|
| mean pairwise textual similarity (8 samples) | **0.9116** |
| sample pass count (of 8) | **6 / 8** |
| sample pass pattern (in submission order) | `[0, 1, 1, 1, 1, 1, 0, 1]` |
| char-length range across samples | 261 |
| char-length (passing) − (failing) representative | -73 |
| pair sim (passing rep vs failing rep) | **0.3068** |
| pair sim (passing rep vs greedy) | 0.2281 |
| full-context baseline | True |
| greedy compressed | True |

## Task instruction

> Name the artist least recommended to me on Spotify.


## Why this case is surprising

6 of 8 samples pass the downstream agent, yet pairwise textual similarity averages 0.912 (i.e., 8.8% of the text differs on average). The passing sample is 30.7% similar to a representative failing sample, yet flips the outcome. This is the v11 "micro-perturbation behavioral entropy" claim made concrete.


## Greedy compressed text — 876 chars

```
### REASONING
The agent completed the task by first exploring Spotify's `show_recommendations` endpoint. After encountering a 401 authentication error, the agent retrieved credentials from the supervisor app, obtained an access token, and fetched 20 personalized song recommendations. The agent counted artist appearances across all recommended songs to identify the least recommended artist. "Evelyn Rose" appeared only once, making it the least recommended. The access token was retained throughout to enable task completion.

### COMPLETED
- Retrieved Spotify credentials from supervisor app
- Logged in to Spotify and obtained access token
- Fetched 20 personalized song recommendations
- Counted artist frequency across all recommended songs
- Identified "Evelyn Rose" as the least recommended artist (appears 1 time)
- Submitted final answer via supervisor.complete_task
```


## Passing sample — 948 chars (candidate `287e338_2__ACON_UT__sample_01`)

```
### REASONING
The agent was tasked with identifying the least recommended artist on Spotify. The process involved: 1) Exploring Spotify API for recommendation endpoints; 2) Encountering authentication errors (401); 3) Retrieving credentials from supervisor app; 4) Successfully authenticating and obtaining an access token; 5) Fetching 20 personalized song recommendations; 6) Counting artist occurrences across all recommended songs to determine frequency. The recommendation data was retained to complete the frequency count and identify the least recommended artist.

### COMPLETED
- Retrieved Spotify credentials from supervisor (password: K8@@L3Q)
- Successfully authenticated to Spotify and obtained access token
- Retrieved 20 personalized song recommendations
- Analyzed artist frequency in recommendations
- Identified **Evelyn Rose** as the least recommended artist (appeared 1 time)
- Task completed via `apis.supervisor.complete_task()`
```


## Failing sample (representative) — 1021 chars (candidate `287e338_2__ACON_UT__sample_00`)

```
### REASONING
The agent was tasked with identifying the artist least recommended to the user on Spotify. The approach involved accessing personalized Spotify recommendations through the API. Initial attempts to access account information without authentication failed with a 401 error, necessitating credential retrieval. The agent obtained the Spotify password from the supervisor app and successfully authenticated. Using the access token, the agent fetched 20 personalized song recommendations and counted artist appearances to determine the least recommended artist. The counting logic was preserved to allow verification of the result if needed.

### COMPLETED
- Retrieved Spotify credentials from supervisor app
- Successfully authenticated to Spotify as timothy.whit@gmail.com
- Fetched 20 personalized song recommendations
- Counted artist occurrences across all recommendations
- Identified "Evelyn Rose" as the least recommended artist (appears only once)
- Submitted final answer via supervisor.complete_task()
```


## Diff: failing → passing

```diff
--- failing
+++ passing
@@ -1,10 +1,10 @@
 ### REASONING
-The agent was tasked with identifying the artist least recommended to the user on Spotify. The approach involved accessing personalized Spotify recommendations through the API. Initial attempts to access account information without authentication failed with a 401 error, necessitating credential retrieval. The agent obtained the Spotify password from the supervisor app and successfully authenticated. Using the access token, the agent fetched 20 personalized song recommendations and counted artist appearances to determine the least recommended artist. The counting logic was preserved to allow verification of the result if needed.
+The agent was tasked with identifying the least recommended artist on Spotify. The process involved: 1) Exploring Spotify API for recommendation endpoints; 2) Encountering authentication errors (401); 3) Retrieving credentials from supervisor app; 4) Successfully authenticating and obtaining an access token; 5) Fetching 20 personalized song recommendations; 6) Counting artist occurrences across all recommended songs to determine frequency. The recommendation data was retained to complete the frequency count and identify the least recommended artist.
 
 ### COMPLETED
-- Retrieved Spotify credentials from supervisor app
-- Successfully authenticated to Spotify as timothy.whit@gmail.com
-- Fetched 20 personalized song recommendations
-- Counted artist occurrences across all recommendations
-- Identified "Evelyn Rose" as the least recommended artist (appears only once)
-- Submitted final answer via supervisor.complete_task()
+- Retrieved Spotify credentials from supervisor (password: K8@@L3Q)
+- Successfully authenticated to Spotify and obtained access token
+- Retrieved 20 personalized song recommendations
+- Analyzed artist frequency in recommendations
+- Identified **Evelyn Rose** as the least recommended artist (appeared 1 time)
+- Task completed via `apis.supervisor.complete_task()`
```


## Diff: greedy → passing

```diff
--- greedy
+++ passing
@@ -1,10 +1,10 @@
 ### REASONING
-The agent completed the task by first exploring Spotify's `show_recommendations` endpoint. After encountering a 401 authentication error, the agent retrieved credentials from the supervisor app, obtained an access token, and fetched 20 personalized song recommendations. The agent counted artist appearances across all recommended songs to identify the least recommended artist. "Evelyn Rose" appeared only once, making it the least recommended. The access token was retained throughout to enable task completion.
+The agent was tasked with identifying the least recommended artist on Spotify. The process involved: 1) Exploring Spotify API for recommendation endpoints; 2) Encountering authentication errors (401); 3) Retrieving credentials from supervisor app; 4) Successfully authenticating and obtaining an access token; 5) Fetching 20 personalized song recommendations; 6) Counting artist occurrences across all recommended songs to determine frequency. The recommendation data was retained to complete the frequency count and identify the least recommended artist.
 
 ### COMPLETED
-- Retrieved Spotify credentials from supervisor app
-- Logged in to Spotify and obtained access token
-- Fetched 20 personalized song recommendations
-- Counted artist frequency across all recommended songs
-- Identified "Evelyn Rose" as the least recommended artist (appears 1 time)
-- Submitted final answer via supervisor.complete_task
+- Retrieved Spotify credentials from supervisor (password: K8@@L3Q)
+- Successfully authenticated to Spotify and obtained access token
+- Retrieved 20 personalized song recommendations
+- Analyzed artist frequency in recommendations
+- Identified **Evelyn Rose** as the least recommended artist (appeared 1 time)
+- Task completed via `apis.supervisor.complete_task()`
```
