# Micro-diff case study — `29caf6f_2` / `general_task_aware` / round `CK`

## Headline numbers

| metric | value |
|---|---|
| mean pairwise textual similarity (8 samples) | **0.8655** |
| sample pass count (of 8) | **5 / 8** |
| sample pass pattern (in submission order) | `[0, 0, 1, 1, 1, 1, 0, 1]` |
| char-length range across samples | 350 |
| char-length (passing) − (failing) representative | -125 |
| pair sim (passing rep vs failing rep) | **0.4944** |
| pair sim (passing rep vs greedy) | 0.5909 |
| full-context baseline | False |
| greedy compressed | True |

## Task instruction

> Laura has asked for my movie recommendations via phone text message. Reply to them with a list of comma-separated movie titles from my Simple Note account as per their request.


## Why this case is surprising

5 of 8 samples pass the downstream agent, yet pairwise textual similarity averages 0.866 (i.e., 13.4% of the text differs on average). The passing sample is 49.4% similar to a representative failing sample, yet flips the outcome. This is the v11 "micro-perturbation behavioral entropy" claim made concrete.


## Greedy compressed text — 733 chars

```
Task: Reply to Laura via phone text with movie recommendations from Simple Note.

Completed:
- Logged into Simple Note (gl.moore@gmail.com / aVfy)Rr)
- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzaW1wbGVfbm90ZStnbC5tb29yZUBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.4N3-gAMhJZOc9-oXaFfJugUPnIytNIdQd3vqPcYOCSA
- Retrieved note_id 1556 ("Movie Recommendations")

Movie list from note: Interstellar, Inception, Black Swan, The Shawshank Redemption, The Wolf of Wall Street, Drive, Her, The Dark Knight Rises, The Big Lebowski, The Dark Knight (and more)

Remaining:
- Phone app password: 2I1#6ld (from supervisor)
- Login to phone app, find Laura's contact, send text with comma-separated movie titles from the list above
```


## Passing sample — 692 chars (candidate `29caf6f_2__general_task_aware__sample_02`)

```
Task: Reply to Laura via phone text with movie recommendations from Simple Note.

Completed:
- Logged into Simple Note (gl.moore@gmail.com)
- Retrieved note "Movie Recommendations" (note_id: 1556)
- Movie list: Interstellar, Inception, Black Swan, The Shawshank Redemption, The Wolf of Wall Street, Drive, Her, The Dark Knight Rises, The Big Lebowski, The Dark Knight (and more)

Failed attempts (do not repeat):
- search_notes without access_token: 401 error
- phone.search_contacts without login: 401 error

Remaining steps:
1. Login to phone app with password: 2I1#6ld
2. Search contacts for "Laura" to get phone number
3. Send text to Laura with comma-separated movie titles from the note
```


## Failing sample (representative) — 817 chars (candidate `29caf6f_2__general_task_aware__sample_01`)

```
Task: Reply to Laura via text with movie recommendations from Simple Note account.

Completed:
- Logged into Simple Note (gl.moore@gmail.com / aVfy)Rr)
- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzaW1wbGVfbm90ZStnbC5tb29yZUBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.4N3-gAMhJZOc9-oXaFfJugUPnIytNIdQd3vqPcYOCSA
- Found "Movie Recommendations" note (note_id: 1556) with movies: Interstellar, Inception, Black Swan, The Shawshank Redemption, The Wolf of Wall Street, Drive, Her, The Dark Knight Rises, The Big Lebowski, The Dark Knight, and more

Failed:
- Phone contacts search returned 401 error - phone app requires authentication

Next steps:
- Authenticate to phone app using password "2I1#6ld"
- Find Laura's phone number
- Send text message to Laura with comma-separated movie titles from the note
```


## Diff: failing → passing

```diff
--- failing
+++ passing
@@ -1,14 +1,15 @@
-Task: Reply to Laura via text with movie recommendations from Simple Note account.
+Task: Reply to Laura via phone text with movie recommendations from Simple Note.
 
 Completed:
-- Logged into Simple Note (gl.moore@gmail.com / aVfy)Rr)
-- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzaW1wbGVfbm90ZStnbC5tb29yZUBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.4N3-gAMhJZOc9-oXaFfJugUPnIytNIdQd3vqPcYOCSA
-- Found "Movie Recommendations" note (note_id: 1556) with movies: Interstellar, Inception, Black Swan, The Shawshank Redemption, The Wolf of Wall Street, Drive, Her, The Dark Knight Rises, The Big Lebowski, The Dark Knight, and more
+- Logged into Simple Note (gl.moore@gmail.com)
+- Retrieved note "Movie Recommendations" (note_id: 1556)
+- Movie list: Interstellar, Inception, Black Swan, The Shawshank Redemption, The Wolf of Wall Street, Drive, Her, The Dark Knight Rises, The Big Lebowski, The Dark Knight (and more)
 
-Failed:
-- Phone contacts search returned 401 error - phone app requires authentication
+Failed attempts (do not repeat):
+- search_notes without access_token: 401 error
+- phone.search_contacts without login: 401 error
 
-Next steps:
-- Authenticate to phone app using password "2I1#6ld"
-- Find Laura's phone number
-- Send text message to Laura with comma-separated movie titles from the note
+Remaining steps:
+1. Login to phone app with password: 2I1#6ld
+2. Search contacts for "Laura" to get phone number
+3. Send text to Laura with comma-separated movie titles from the note
```


## Diff: greedy → passing

```diff
--- greedy
+++ passing
@@ -2,11 +2,14 @@
 
 Completed:
-- Logged into Simple Note (gl.moore@gmail.com / aVfy)Rr)
-- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzaW1wbGVfbm90ZStnbC5tb29yZUBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.4N3-gAMhJZOc9-oXaFfJugUPnIytNIdQd3vqPcYOCSA
-- Retrieved note_id 1556 ("Movie Recommendations")
+- Logged into Simple Note (gl.moore@gmail.com)
+- Retrieved note "Movie Recommendations" (note_id: 1556)
+- Movie list: Interstellar, Inception, Black Swan, The Shawshank Redemption, The Wolf of Wall Street, Drive, Her, The Dark Knight Rises, The Big Lebowski, The Dark Knight (and more)
 
-Movie list from note: Interstellar, Inception, Black Swan, The Shawshank Redemption, The Wolf of Wall Street, Drive, Her, The Dark Knight Rises, The Big Lebowski, The Dark Knight (and more)
+Failed attempts (do not repeat):
+- search_notes without access_token: 401 error
+- phone.search_contacts without login: 401 error
 
-Remaining:
-- Phone app password: 2I1#6ld (from supervisor)
-- Login to phone app, find Laura's contact, send text with comma-separated movie titles from the list above
+Remaining steps:
+1. Login to phone app with password: 2I1#6ld
+2. Search contacts for "Laura" to get phone number
+3. Send text to Laura with comma-separated movie titles from the note
```
