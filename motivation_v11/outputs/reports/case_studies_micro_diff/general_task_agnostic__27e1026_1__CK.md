# Micro-diff case study — `27e1026_1` / `general_task_agnostic` / round `CK`

## Headline numbers

| metric | value |
|---|---|
| mean pairwise textual similarity (8 samples) | **0.8424** |
| sample pass count (of 8) | **7 / 8** |
| sample pass pattern (in submission order) | `[1, 1, 1, 0, 1, 1, 1, 1]` |
| char-length range across samples | 209 |
| char-length (passing) − (failing) representative | -72 |
| pair sim (passing rep vs failing rep) | **0.4508** |
| pair sim (passing rep vs greedy) | 0.4034 |
| full-context baseline | False |
| greedy compressed | False |

## Task instruction

> What is the title of the oldest released song in my Spotify account from across my song, album and playlist libraries?


## Why this case is surprising

7 of 8 samples pass the downstream agent, yet pairwise textual similarity averages 0.842 (i.e., 15.8% of the text differs on average). The passing sample is 45.1% similar to a representative failing sample, yet flips the outcome. This is the v11 "micro-perturbation behavioral entropy" claim made concrete.


## Greedy compressed text — 333 chars

```
Goal: Find oldest released song in Spotify account.

Steps:
- Retrieved credentials from supervisor
- Retrieved song library, album library, playlist library
- Collected all unique song IDs
- Queried each song for release_date
- Sorted by release_date

Result: Oldest song is "Final Act" (song_id: 309), released 2019-07-02T06:55:50.
```


## Passing sample — 381 chars (candidate `27e1026_1__general_task_agnostic__sample_00`)

```
Goal: Find oldest released song in Spotify account.

Credentials: erikabail@gmail.com, password XZ!zW*T

Process:
- Login to get access token
- Retrieved song, album, and playlist libraries
- Collected song IDs from libraries, albums (2, 6, 10, 11), playlists (500-504)
- Queried show_song API for release_date

Result: "Final Act" (song_id: 309), release_date: 2019-07-02T06:55:50
```


## Failing sample (representative) — 453 chars (candidate `27e1026_1__general_task_agnostic__sample_03`)

```
Goal: Find oldest released song in Spotify account.

Credentials: erikabail@gmail.com / XZ!zW*T

Process:
- Retrieved song, album, and playlist libraries
- Collected all unique song IDs
- Called show_song for each ID to get release_date
- Sorted by release_date ascending

Result: "Final Act" (song_id 309), released 2019-07-02T06:55:50

Technical note: Spotify API access_token must be passed explicitly to each API call; doesn't persist between calls.
```


## Diff: failing → passing

```diff
--- failing
+++ passing
@@ -1,13 +1,11 @@
 Goal: Find oldest released song in Spotify account.
 
-Credentials: erikabail@gmail.com / XZ!zW*T
+Credentials: erikabail@gmail.com, password XZ!zW*T
 
 Process:
+- Login to get access token
 - Retrieved song, album, and playlist libraries
-- Collected all unique song IDs
-- Called show_song for each ID to get release_date
-- Sorted by release_date ascending
+- Collected song IDs from libraries, albums (2, 6, 10, 11), playlists (500-504)
+- Queried show_song API for release_date
 
-Result: "Final Act" (song_id 309), released 2019-07-02T06:55:50
-
-Technical note: Spotify API access_token must be passed explicitly to each API call; doesn't persist between calls.
+Result: "Final Act" (song_id: 309), release_date: 2019-07-02T06:55:50
```


## Diff: greedy → passing

```diff
--- greedy
+++ passing
@@ -1,10 +1,11 @@
 Goal: Find oldest released song in Spotify account.
 
-Steps:
-- Retrieved credentials from supervisor
-- Retrieved song library, album library, playlist library
-- Collected all unique song IDs
-- Queried each song for release_date
-- Sorted by release_date
+Credentials: erikabail@gmail.com, password XZ!zW*T
 
-Result: Oldest song is "Final Act" (song_id: 309), released 2019-07-02T06:55:50.
+Process:
+- Login to get access token
+- Retrieved song, album, and playlist libraries
+- Collected song IDs from libraries, albums (2, 6, 10, 11), playlists (500-504)
+- Queried show_song API for release_date
+
+Result: "Final Act" (song_id: 309), release_date: 2019-07-02T06:55:50
```
