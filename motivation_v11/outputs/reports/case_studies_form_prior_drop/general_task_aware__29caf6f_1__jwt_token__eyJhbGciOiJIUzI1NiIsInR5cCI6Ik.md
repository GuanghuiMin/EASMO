# Form-prior drop case — `29caf6f_1` / `general_task_aware` / round `CK`

## Token under analysis

- **category**: `jwt_token`
- **value (truncated)**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzaW1wbGVfbm9…`
- **original-trajectory context**:
  > ...rname='joyce-weav@gmail.com', password='RluCyXn'))  output: {  "access_token": "**eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzaW1wbGVfbm90ZStqb3ljZS13ZWF2QGdtYWlsLmNvbSIsImV4cCI6MTY4NDQxMjA5OH0.ylS_T2qEp3N0QmX7hyfNwXb-VcLzKeReZ9HRKhmVJ-U**",  "token_type": "Bearer" }  ### step 9 action: # Now let me search for notes w...

## Headline numbers

| metric | value |
|---|---|
| sample preservation count | **4 / 8** |
| pass rate WITH token preserved  | **100%** |
| pass rate WITHOUT token (dropped) | **50%** |
| **pass-rate gap (with − without)** | **+50 pp** |
| greedy preserved this token? | False |
| greedy pass | True |
| full-context baseline | False |
| sample preserve pattern (submission order) | `[0, 1, 0, 0, 1, 1, 0, 1]` |
| sample pass pattern | `[1, 1, 1, 0, 1, 1, 0, 1]` |

## Task instruction

> Christopher has asked for my movie recommendations via phone text message. Reply to them with a list of comma-separated movie titles from my Simple Note account as per their request.


## v5 / v7 narrative for this case

The token `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzaW1wbGVfbm9…` is present in the original AppWorld trajectory; under K=2 recompression in the `general_task_aware` family, **4 of 8 stochastic samples drop it**, while 4 preserve it. Samples that preserve the token pass downstream at 100%; samples that drop it pass at 50% (+50 pp gap). This is the v5 'recovered-then-dropped' phenomenon and the v7 'form-prior conditions on fact_type, not need_label' phenomenon manifesting at sample-distribution level: the same compressor applied to the same input drops critical functional content with non-trivial probability, and best-of-N selection exploits this variance.


## Greedy compressed — 550 chars (candidate `29caf6f_1__general_task_aware__greedy`)

- token `jwt_token` preserved: **False**

```
Task: Reply to Christopher via phone text with movie recommendations from Simple Note.

Completed:
- Logged into Simple Note and retrieved note "Movie Recommendations" (note_id: 1608)
- Movies to send: Interstellar, Forrest Gump, Blade Runner 2049, No Country for Old Men, Schindler's List, The Revenant, The Wolf of Wall Street, The Green Mile, Fight Club

Blocked:
- Phone API returned 401 error - authentication required

Next steps:
- Login to phone app using password: QAEZ+BF
- Find Christopher's contact
- Send text message with the movie list
```

## Passing sample (token preserved) — 970 chars (candidate `29caf6f_1__general_task_aware__sample_01`)

- token `jwt_token` preserved: **True**
- snippet around token: > ...Simple Note (username: joyce-weav@gmail.com, password: RluCyXn) - Access token: **eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzaW1wbGVfbm90ZStqb3ljZS13ZWF2QGdtYWlsLmNvbSIsImV4cCI6MTY4NDQxMjA5OH0.ylS_T2qEp3N0QmX7hyfNwXb-VcLzKeReZ9HRKhmVJ-U** - Retrieved note "Movie Recommendations" (note_id: 1608) containing: Interstell...

```
Task: Reply to Christopher via phone text message with movie recommendations from Simple Note.

Completed:
- Logged into Simple Note (username: joyce-weav@gmail.com, password: RluCyXn)
- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzaW1wbGVfbm90ZStqb3ljZS13ZWF2QGdtYWlsLmNvbSIsImV4cCI6MTY4NDQxMjA5OH0.ylS_T2qEp3N0QmX7hyfNwXb-VcLzKeReZ9HRKhmVJ-U
- Retrieved note "Movie Recommendations" (note_id: 1608) containing: Interstellar, Forrest Gump, Blade Runner 2049, No Country for Old Men, Schindler's List, The Revenant, The Green Mile, Fight Club (content truncated)

Next steps:
1. Login to phone app (email: joyce-weav@gmail.com, password: QAEZ+BF)
2. Search for Christopher's contact
3. Send text with comma-separated movie list

Failed attempts (do not repeat):
- search_notes without access_token → 401 error
- simple_note.login with 'email' param instead of 'username' → 422 validation error
- phone.search_contacts without authentication → 401 error
```

## Failing sample (token dropped, representative) — 772 chars (candidate `29caf6f_1__general_task_aware__sample_03`)

- token `jwt_token` preserved: **False**

```
Task: Reply to Christopher via phone text with movie recommendations from Simple Note.

Completed:
- Logged into Simple Note (username: joyce-weav@gmail.com)
- Retrieved note "Movie Recommendations" (note_id: 1608) with movies: Interstellar, Forrest Gump, Blade Runner 2049, No Country for Old Men, Schindler's List, The Revenant, The Wolf of Wall Street, The Green Mile, Fight Club (content may be truncated)

Remaining steps:
1. Login to phone app using supervisor account (email: joyce-weav@gmail.com, password: QAEZ+BF)
2. Search for Christopher's contact
3. Send text with comma-separated movie titles

Failed attempts (avoid repeating):
- search_notes failed with 401 - access_token wasn't included
- phone search_contacts failed with 401 - not logged into phone app
```