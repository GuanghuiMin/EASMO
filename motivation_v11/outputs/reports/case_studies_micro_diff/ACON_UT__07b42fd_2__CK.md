# Micro-diff case study — `07b42fd_2` / `ACON_UT` / round `CK`

## Headline numbers

| metric | value |
|---|---|
| mean pairwise textual similarity (8 samples) | **0.9152** |
| sample pass count (of 8) | **5 / 8** |
| sample pass pattern (in submission order) | `[0, 1, 1, 1, 0, 1, 1, 0]` |
| char-length range across samples | 204 |
| char-length (passing) − (failing) representative | -122 |
| pair sim (passing rep vs failing rep) | **0.2288** |
| pair sim (passing rep vs greedy) | 0.1871 |
| full-context baseline | True |
| greedy compressed | False |

## Task instruction

> Follow all the edm artists on Spotify that have at least 23 followers.


## Why this case is surprising

5 of 8 samples pass the downstream agent, yet pairwise textual similarity averages 0.915 (i.e., 8.5% of the text differs on average). The passing sample is 22.9% similar to a representative failing sample, yet flips the outcome. This is the v11 "micro-perturbation behavioral entropy" claim made concrete.


## Greedy compressed text — 855 chars

```
### REASONING

The agent successfully completed the task of following EDM artists with at least 23 followers. The search_artists API was used with genre="edm" and min_follower_count=23 parameters. Only one artist (Jasper Skye) met the criteria. The agent obtained Spotify credentials from the supervisor, logged in to get an access token, and successfully followed the artist. The access token and credentials were retained throughout but the task is now complete.

### COMPLETED

- Searched for EDM artists with ≥23 followers using search_artists API
- Identified 1 qualifying artist: Jasper Skye (artist_id: 3) with 23 followers
- Retrieved Spotify credentials from supervisor (password: i5ZKWxI)
- Logged in to Spotify and obtained access token
- Successfully followed artist Jasper Skye via follow_artist API
- Task completed with answer "Jasper Skye"
```


## Passing sample — 1058 chars (candidate `07b42fd_2__ACON_UT__sample_01`)

```
### REASONING
The agent completed the task by systematically exploring available Spotify APIs to identify the correct tools. The search_artists API was selected as the appropriate method, requiring genre and minimum_follower_count parameters. Since follow_artist requires authentication, the agent retrieved Spotify credentials from the supervisor app (password: i5ZKWxI) and logged in to obtain an access token. The search revealed only one artist meeting the criteria (EDM genre with ≥23 followers), so a single follow action was sufficient. The password was retained in case re-authentication is needed for future Spotify operations.

### COMPLETED
- Explored Spotify APIs: search_artists, follow_artist, login
- Retrieved Spotify credentials from supervisor (password: i5ZKWxI)
- Authenticated to Spotify and obtained access token
- Searched for EDM artists with min_follower_count=23
- Found 1 qualifying artist: Jasper Skye (artist_id: 3) with exactly 23 followers
- Successfully followed artist Jasper Skye
- Task completed with response "Jasper Skye"
```


## Failing sample (representative) — 1180 chars (candidate `07b42fd_2__ACON_UT__sample_00`)

```
### REASONING

The agent systematically explored Spotify APIs to accomplish the task of following EDM artists with at least 23 followers. The available APIs were examined: `search_artists` for finding artists by genre and follower count, and `follow_artist` for following them. The search API confirmed it supports filtering by genre ("edm") and minimum follower count parameters. After retrieving the Spotify password from the supervisor app, the agent logged in to obtain an access token required for the follow operation. The search on page 0 returned only one qualifying artist (Jasper Skye with exactly 23 followers), indicating this was likely the complete set of results. The agent then successfully followed this artist and marked the task complete.

### COMPLETED

- Searched Spotify for EDM artists with ≥23 followers using `search_artists` API
- Found only one qualifying artist: **Jasper Skye** (artist_id: 3) with 23 followers
- Retrieved Spotify credentials from supervisor (password: i5ZKWxI)
- Logged in to Spotify to obtain access token
- Successfully followed Jasper Skye via `follow_artist` API
- Task completed with answer "Jasper Skye" submitted to supervisor
```


## Diff: failing → passing

```diff
--- failing
+++ passing
@@ -1,12 +1,11 @@
 ### REASONING
-
-The agent systematically explored Spotify APIs to accomplish the task of following EDM artists with at least 23 followers. The available APIs were examined: `search_artists` for finding artists by genre and follower count, and `follow_artist` for following them. The search API confirmed it supports filtering by genre ("edm") and minimum follower count parameters. After retrieving the Spotify password from the supervisor app, the agent logged in to obtain an access token required for the follow operation. The search on page 0 returned only one qualifying artist (Jasper Skye with exactly 23 followers), indicating this was likely the complete set of results. The agent then successfully followed this artist and marked the task complete.
+The agent completed the task by systematically exploring available Spotify APIs to identify the correct tools. The search_artists API was selected as the appropriate method, requiring genre and minimum_follower_count parameters. Since follow_artist requires authentication, the agent retrieved Spotify credentials from the supervisor app (password: i5ZKWxI) and logged in to obtain an access token. The search revealed only one artist meeting the criteria (EDM genre with ≥23 followers), so a single follow action was sufficient. The password was retained in case re-authentication is needed for future Spotify operations.
 
 ### COMPLETED
-
-- Searched Spotify for EDM artists with ≥23 followers using `search_artists` API
-- Found only one qualifying artist: **Jasper Skye** (artist_id: 3) with 23 followers
+- Explored Spotify APIs: search_artists, follow_artist, login
 - Retrieved Spotify credentials from supervisor (password: i5ZKWxI)
-- Logged in to Spotify to obtain access token
-- Successfully followed Jasper Skye via `follow_artist` API
-- Task completed with answer "Jasper Skye" submitted to supervisor
+- Authenticated to Spotify and obtained access token
+- Searched for EDM artists with min_follower_count=23
+- Found 1 qualifying artist: Jasper Skye (artist_id: 3) with exactly 23 followers
+- Successfully followed artist Jasper Skye
+- Task completed with response "Jasper Skye"
```


## Diff: greedy → passing

```diff
--- greedy
+++ passing
@@ -1,12 +1,11 @@
 ### REASONING
-
-The agent successfully completed the task of following EDM artists with at least 23 followers. The search_artists API was used with genre="edm" and min_follower_count=23 parameters. Only one artist (Jasper Skye) met the criteria. The agent obtained Spotify credentials from the supervisor, logged in to get an access token, and successfully followed the artist. The access token and credentials were retained throughout but the task is now complete.
+The agent completed the task by systematically exploring available Spotify APIs to identify the correct tools. The search_artists API was selected as the appropriate method, requiring genre and minimum_follower_count parameters. Since follow_artist requires authentication, the agent retrieved Spotify credentials from the supervisor app (password: i5ZKWxI) and logged in to obtain an access token. The search revealed only one artist meeting the criteria (EDM genre with ≥23 followers), so a single follow action was sufficient. The password was retained in case re-authentication is needed for future Spotify operations.
 
 ### COMPLETED
-
-- Searched for EDM artists with ≥23 followers using search_artists API
-- Identified 1 qualifying artist: Jasper Skye (artist_id: 3) with 23 followers
+- Explored Spotify APIs: search_artists, follow_artist, login
 - Retrieved Spotify credentials from supervisor (password: i5ZKWxI)
-- Logged in to Spotify and obtained access token
-- Successfully followed artist Jasper Skye via follow_artist API
-- Task completed with answer "Jasper Skye"
+- Authenticated to Spotify and obtained access token
+- Searched for EDM artists with min_follower_count=23
+- Found 1 qualifying artist: Jasper Skye (artist_id: 3) with exactly 23 followers
+- Successfully followed artist Jasper Skye
+- Task completed with response "Jasper Skye"
```
