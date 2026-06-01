# Form-prior drop case — `7d7fbf6_2` / `general_task_agnostic` / round `CK`

## Token under analysis

- **category**: `jwt_token`
- **value (truncated)**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3R…`
- **original-trajectory context**:
  > ...ername='jasonsimp@gmail.com', password='6ubYf]C'))  output: {  "access_token": "**eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStqYXNvbnNpbXBAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ._cqOYHfZ0nZ345YW72q2I8SdweuWDc6b4okDY-gma90**",  "token_type": "Bearer" }  ### step 8 action: # Now let me explore the photos...

## Headline numbers

| metric | value |
|---|---|
| sample preservation count | **3 / 8** |
| pass rate WITH token preserved  | **100%** |
| pass rate WITHOUT token (dropped) | **20%** |
| **pass-rate gap (with − without)** | **+80 pp** |
| greedy preserved this token? | False |
| greedy pass | True |
| full-context baseline | False |
| sample preserve pattern (submission order) | `[0, 0, 0, 0, 1, 1, 0, 1]` |
| sample pass pattern | `[0, 0, 1, 0, 1, 1, 0, 1]` |

## Task instruction

> The "~/photos/" directory in my file system has photo files organized in sub-directories for each vacation spot. Compress them and save them in "~/photos/vacations/<vacation_spot>.tar" for each vacation spot, and then delete all vacation spot sub-directories. Here, <vacation_spot> is the name of the vacation spot as it appears in the sub-directory name.


## v5 / v7 narrative for this case

The token `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3R…` is present in the original AppWorld trajectory; under K=2 recompression in the `general_task_agnostic` family, **5 of 8 stochastic samples drop it**, while 3 preserve it. Samples that preserve the token pass downstream at 100%; samples that drop it pass at 20% (+80 pp gap). This is the v5 'recovered-then-dropped' phenomenon and the v7 'form-prior conditions on fact_type, not need_label' phenomenon manifesting at sample-distribution level: the same compressor applied to the same input drops critical functional content with non-trivial probability, and best-of-N selection exploits this variance.


## Greedy compressed — 407 chars (candidate `7d7fbf6_2__general_task_agnostic__greedy`)

- token `jwt_token` preserved: **False**

```
- File system API: login required (Jason Simpson, jasonsimp@gmail.com, password: 6ubYf]C)
- Logged in, obtained access token
- ~/photos/vacations/: bali/, barcelona/, kyoto/, maldives/
- compress_directory API: requires access_token, directory_path, compressed_file_path, delete_directory, overwrite
- Compressed bali/ → bali.tar (deleted original)
- Compressed barcelona/ → barcelona.tar (deleted original)
```

## Passing sample (token preserved) — 499 chars (candidate `7d7fbf6_2__general_task_agnostic__sample_04`)

- token `jwt_token` preserved: **True**
- snippet around token: > ... barcelona.tar  Remaining: kyoto/ and maldives/ need compression  Access token: **eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStqYXNvbnNpbXBAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ._cqOYHfZ0nZ345YW72q2I8SdweuWDc6b4okDY-gma90**...

```
User: Jason Simpson (jasonsimp@gmail.com)

Goal: Compress vacation directories in ~/photos/vacations/

Completed:
- Logged in, obtained access_token
- Found 4 directories: bali, barcelona, kyoto, maldives
- Compressed bali/ → bali.tar
- Compressed barcelona/ → barcelona.tar

Remaining: kyoto/ and maldives/ need compression

Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStqYXNvbnNpbXBAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ._cqOYHfZ0nZ345YW72q2I8SdweuWDc6b4okDY-gma90
```

## Failing sample (token dropped, representative) — 560 chars (candidate `7d7fbf6_2__general_task_agnostic__sample_00`)

- token `jwt_token` preserved: **False**

```
- Got file_system password (6ubYf]C) from supervisor.show_account_passwords()
- Profile: Jason Simpson, jasonsimp@gmail.com
- Logged into file_system, obtained access token
- Explored ~/photos/: found vacation_bali.zip and vacations/ subdirectory
- ~/photos/vacations/: bali/, barcelona/, kyoto/, maldives/
- compress_directory API: requires access_token, directory_path, compressed_file_path, delete_directory, overwrite
- Compressed bali/ → bali.tar (deleted original)
- Compressed barcelona/ → barcelona.tar (deleted original)
- Remaining: kyoto/, maldives/
```