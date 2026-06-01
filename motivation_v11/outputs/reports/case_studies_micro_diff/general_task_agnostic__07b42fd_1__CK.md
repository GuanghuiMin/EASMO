# Micro-diff case study — `07b42fd_1` / `general_task_agnostic` / round `CK`

## Headline numbers

| metric | value |
|---|---|
| mean pairwise textual similarity (8 samples) | **0.8475** |
| sample pass count (of 8) | **5 / 8** |
| sample pass pattern (in submission order) | `[0, 0, 1, 1, 1, 1, 1, 0]` |
| char-length range across samples | 194 |
| char-length (passing) − (failing) representative | +107 |
| pair sim (passing rep vs failing rep) | **0.6575** |
| pair sim (passing rep vs greedy) | 0.5408 |
| full-context baseline | True |
| greedy compressed | True |

## Task instruction

> Follow all the classical artists on Spotify that have at least 22 followers.


## Why this case is surprising

5 of 8 samples pass the downstream agent, yet pairwise textual similarity averages 0.848 (i.e., 15.2% of the text differs on average). The passing sample is 65.8% similar to a representative failing sample, yet flips the outcome. This is the v11 "micro-perturbation behavioral entropy" claim made concrete.


## Greedy compressed text — 320 chars

```
- Goal: Find classical artists on Spotify with ≥22 followers and follow them
- Found Grace Harrison (artist_id=14) with exactly 22 followers
- Logged in, got access token
- First follow_artist call failed (401 - missing access_token)
- Called follow_artist with artist_id=14 and access_token - succeeded
- Task completed
```


## Passing sample — 379 chars (candidate `07b42fd_1__general_task_agnostic__sample_04`)

```
- Goal: Find classical artists on Spotify with ≥22 followers and follow them
- Searched "classical": found 5 artists, only Grace Harrison (artist_id=14) met criteria with 22 followers
- Follow failed (401 error - needed login)
- Logged in with credentials: eri_powe@gmail.com / IXSwolL
- Obtained access token, successfully followed Grace Harrison (artist_id=14)
- Task completed
```


## Failing sample (representative) — 272 chars (candidate `07b42fd_1__general_task_agnostic__sample_00`)

```
Goal: Find classical artists on Spotify with ≥22 followers and follow them. Found 5 classical artists - only Grace Harrison (artist_id=14) has exactly 22 followers. Logged in with credentials eri_powe@gmail.com / IXSwolL. Successfully followed artist_id=14. Task complete.
```


## Diff: failing → passing

```diff
--- failing
+++ passing
@@ -1 +1,6 @@
-Goal: Find classical artists on Spotify with ≥22 followers and follow them. Found 5 classical artists - only Grace Harrison (artist_id=14) has exactly 22 followers. Logged in with credentials eri_powe@gmail.com / IXSwolL. Successfully followed artist_id=14. Task complete.
+- Goal: Find classical artists on Spotify with ≥22 followers and follow them
+- Searched "classical": found 5 artists, only Grace Harrison (artist_id=14) met criteria with 22 followers
+- Follow failed (401 error - needed login)
+- Logged in with credentials: eri_powe@gmail.com / IXSwolL
+- Obtained access token, successfully followed Grace Harrison (artist_id=14)
+- Task completed
```


## Diff: greedy → passing

```diff
--- greedy
+++ passing
@@ -1,6 +1,6 @@
 - Goal: Find classical artists on Spotify with ≥22 followers and follow them
-- Found Grace Harrison (artist_id=14) with exactly 22 followers
-- Logged in, got access token
-- First follow_artist call failed (401 - missing access_token)
-- Called follow_artist with artist_id=14 and access_token - succeeded
+- Searched "classical": found 5 artists, only Grace Harrison (artist_id=14) met criteria with 22 followers
+- Follow failed (401 error - needed login)
+- Logged in with credentials: eri_powe@gmail.com / IXSwolL
+- Obtained access token, successfully followed Grace Harrison (artist_id=14)
 - Task completed
```
