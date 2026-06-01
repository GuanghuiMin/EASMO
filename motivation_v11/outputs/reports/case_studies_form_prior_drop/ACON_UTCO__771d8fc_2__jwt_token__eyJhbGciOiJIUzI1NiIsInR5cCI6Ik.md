# Form-prior drop case — `771d8fc_2` / `ACON_UTCO` / round `CK`

## Token under analysis

- **category**: `jwt_token`
- **value (truncated)**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NTA…`
- **original-trajectory context**:
  > ....login(username='9503658964', password='ox6SqF4'))  output: {  "access_token": "**eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NTAzNjU4OTY0IiwiZXhwIjoxNjg0NDEyMDk4fQ.WIoBUqmPNoG2I3zrIsPlXAQaM4dx4GLylGtdOdR14ow**",  "token_type": "Bearer" }  ### step 7 action: # Now let me search for text me...

## Headline numbers

| metric | value |
|---|---|
| sample preservation count | **3 / 8** |
| pass rate WITH token preserved  | **100%** |
| pass rate WITHOUT token (dropped) | **20%** |
| **pass-rate gap (with − without)** | **+80 pp** |
| greedy preserved this token? | False |
| greedy pass | True |
| full-context baseline | True |
| sample preserve pattern (submission order) | `[0, 0, 0, 1, 1, 0, 1, 0]` |
| sample pass pattern | `[0, 1, 0, 1, 1, 0, 1, 0]` |

## Task instruction

> All phone text messages and voice messages from 9294880327 are spam, delete them.


## v5 / v7 narrative for this case

The token `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NTA…` is present in the original AppWorld trajectory; under K=2 recompression in the `ACON_UTCO` family, **5 of 8 stochastic samples drop it**, while 3 preserve it. Samples that preserve the token pass downstream at 100%; samples that drop it pass at 20% (+80 pp gap). This is the v5 'recovered-then-dropped' phenomenon and the v7 'form-prior conditions on fact_type, not need_label' phenomenon manifesting at sample-distribution level: the same compressor applied to the same input drops critical functional content with non-trivial probability, and best-of-N selection exploits this variance.


## Greedy compressed — 1151 chars (candidate `771d8fc_2__ACON_UTCO__greedy`)

- token `jwt_token` preserved: **False**

```
### REASONING
The agent was tasked with deleting all text and voice messages from phone number 9294880327 (identified as spam). The agent completed the full workflow: explored available apps, identified the phone app, retrieved login credentials from the supervisor, logged in, searched for both text and voice messages from the target number, found 11 text messages and 8 voice messages, and systematically deleted all of them using the appropriate delete APIs. The task was marked complete via supervisor.complete_task().

### COMPLETED
- Logged into phone app with username 9503658964 and password ox6SqF4
- Found and deleted 11 text messages from 9294880327 (IDs: 16802, 16796, 16810, 16814, 16808, 16806, 16804, 16800, 16798, 16794, 16812)
- Found and deleted 8 voice messages from 9294880327 (IDs: 17028, 17036, 17026, 17030, 17022, 17024, 17034, 17032)
- Task completed successfully via supervisor.complete_task()

### STATE RETAINED
- **already_processed_song_ids**: N/A (not a music task)
- **current_highest_liked_song**: N/A
- **processing_progress**: N/A - task fully completed
- **cached_api_results**: N/A
- **early_exit_threshold**: N/A
```

## Passing sample (token preserved) — 1635 chars (candidate `771d8fc_2__ACON_UTCO__sample_02`)

- token `jwt_token` preserved: **True**
- snippet around token: > ...: [17028, 17036, 17026, 17030, 17022, 17024, 17034, 17032] - **access_token**: "**eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NTAzNjU4OTY0IiwiZXhwIjoxNjg0NDEyMDk4fQ.WIoBUqmPNoG2I3zrIsPlXAQaM4dx4GLylGtdOdR14ow**" - **task_status**: COMPLETE...

```
### REASONING

The agent successfully completed the spam message deletion task. The approach was systematic:
1. Explored available apps to locate the phone app
2. Obtained login credentials from supervisor (username: 9503658964, password: ox6SqF4)
3. Authenticated and obtained Bearer token for API access
4. Searched for text messages from 9294880327 → found 11 messages
5. Searched for voice messages from 9294880327 → found 8 messages
6. Deleted all 11 text messages using their IDs
7. Deleted all 8 voice messages using their IDs
8. Marked task complete via supervisor.complete_task()

All deletion operations succeeded. The task is fully completed with no remaining work.

### COMPLETED

- **Login to phone app**: Successfully authenticated with supervisor credentials, obtained Bearer token
- **Text message deletion**: Deleted all 11 text messages from 9294880327 (IDs: 16802, 16796, 16810, 16814, 16808, 16806, 16804, 16800, 16798, 16794, 16812)
- **Voice message deletion**: Deleted all 8 voice messages from 9294880327 (IDs: 17028, 17036, 17026, 17030, 17022, 17024, 17034, 17032)
- **Task completion**: Called supervisor.complete_task() - execution successful

### STATE RETAINED

- **target_phone_number**: "9294880327" (spam number)
- **deleted_text_message_ids**: [16802, 16796, 16810, 16814, 16808, 16806, 16804, 16800, 16798, 16794, 16812]
- **deleted_voice_message_ids**: [17028, 17036, 17026, 17030, 17022, 17024, 17034, 17032]
- **access_token**: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NTAzNjU4OTY0IiwiZXhwIjoxNjg0NDEyMDk4fQ.WIoBUqmPNoG2I3zrIsPlXAQaM4dx4GLylGtdOdR14ow"
- **task_status**: COMPLETE
```

## Failing sample (token dropped, representative) — 1347 chars (candidate `771d8fc_2__ACON_UTCO__sample_00`)

- token `jwt_token` preserved: **False**

```
### REASONING
The agent successfully completed the spam message deletion task. It first explored available apps to identify the phone app as the relevant tool, retrieved login credentials from the supervisor, and authenticated successfully. The agent then searched for both text and voice messages from the spam number 9294880327, finding 11 text messages and 8 voice messages. Using the delete APIs, all messages were removed. The task was marked as complete via supervisor.complete_task(). No errors were encountered during execution.

### COMPLETED
- **Login to phone app**: Successfully authenticated using username '9503658964' and password from supervisor
- **Text message deletion**: Deleted all 11 text messages from 9294880327 (IDs: 16802, 16796, 16810, 16814, 16808, 16806, 16804, 16800, 16798, 16794, 16812)
- **Voice message deletion**: Deleted all 8 voice messages from 9294880327 (IDs: 17028, 17036, 17026, 17030, 17022, 17024, 17034, 17032)
- **Task completion**: Called supervisor.complete_task() - execution successful

### STATE RETAINED
- **already_processed_song_ids**: Not applicable (message deletion task, not song processing)
- **current_highest_liked_song**: Not applicable
- **processing_progress**: Not applicable (task fully completed)
- **cached_api_results**: Not applicable
- **early_exit_threshold**: Not applicable
```