# Form-prior drop case — `7d7fbf6_3` / `general_task_agnostic` / round `CK`

## Token under analysis

- **category**: `kv_pair`
- **value (truncated)**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3R…`
- **original-trajectory context**:
  > ...name='adrian_burt@gmail.com', password='unJp1*f'))  output: {  "access_token": "**eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbSthZHJpYW5fYnVydEBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.VxszQcGCTa0saz1mg8gYsS1PGhw78OvKLvhvLwBummg**",  "token_type": "Bearer" }  ### step 8 action: # Now let me see what's in the ...

## Headline numbers

| metric | value |
|---|---|
| sample preservation count | **5 / 8** |
| pass rate WITH token preserved  | **60%** |
| pass rate WITHOUT token (dropped) | **0%** |
| **pass-rate gap (with − without)** | **+60 pp** |
| greedy preserved this token? | False |
| greedy pass | False |
| full-context baseline | False |
| sample preserve pattern (submission order) | `[0, 1, 1, 1, 1, 0, 0, 1]` |
| sample pass pattern | `[0, 0, 0, 1, 1, 0, 0, 1]` |

## Task instruction

> The "~/pictures/" directory in my file system has photo files organized in sub-directories for each vacation spot. Compress them and save them in "~/pictures/vacations/<vacation_spot>.zip" for each vacation spot, and then delete all vacation spot sub-directories. Here, <vacation_spot> is the name of the vacation spot as it appears in the sub-directory name.


## v5 / v7 narrative for this case

The token `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3R…` is present in the original AppWorld trajectory; under K=2 recompression in the `general_task_agnostic` family, **3 of 8 stochastic samples drop it**, while 5 preserve it. Samples that preserve the token pass downstream at 60%; samples that drop it pass at 0% (+60 pp gap). This is the v5 'recovered-then-dropped' phenomenon and the v7 'form-prior conditions on fact_type, not need_label' phenomenon manifesting at sample-distribution level: the same compressor applied to the same input drops critical functional content with non-trivial probability, and best-of-N selection exploits this variance.


## Greedy compressed — 304 chars (candidate `7d7fbf6_3__general_task_agnostic__greedy`)

- token `kv_pair` preserved: **False**

```
- Accessed file_system API for directory/compression functions
- Logged in: adrian_burt@gmail.com, password: "unJp1*f" (access token obtained)
- Listed ~/pictures/: bali, maui, santorini, seoul
- Compressed bali, maui, santorini to zip using compress_directory API (delete_directory=True, overwrite=True)
```

## Passing sample (token preserved) — 383 chars (candidate `7d7fbf6_3__general_task_agnostic__sample_04`)

- token `kv_pair` preserved: **True**
- snippet around token: > ...ictures/vacations/ to zip files - Auth: adrian_burt@gmail.com / unJp1*f, token: **eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbSthZHJpYW5fYnVydEBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.VxszQcGCTa0saz1mg8gYsS1PGhw78OvKLvhvLwBummg** - Found: bali, maui, santorini, seoul - Compressed: bali.zip, maui.zip, santori...

```
- Task: Compress vacation dirs in ~/pictures/vacations/ to zip files
- Auth: adrian_burt@gmail.com / unJp1*f, token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbSthZHJpYW5fYnVydEBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.VxszQcGCTa0saz1mg8gYsS1PGhw78OvKLvhvLwBummg
- Found: bali, maui, santorini, seoul
- Compressed: bali.zip, maui.zip, santorini.zip
- Remaining: seoul
```

## Failing sample (token dropped, representative) — 301 chars (candidate `7d7fbf6_3__general_task_agnostic__sample_00`)

- token `kv_pair` preserved: **False**

```
Task: Compress vacation directories in ~/pictures/vacations/ to zip files

- Logged into file_system with email adrian_burt@gmail.com (access token)
- Found 4 vacation sub-directories: bali, maui, santorini, seoul
- Compressed bali, maui, santorini to zip files (delete_directory=True, overwrite=True)
```