# Form-prior drop case — `302c169_3` / `general_task_agnostic` / round `CK`

## Token under analysis

- **category**: `jwt_token`
- **value (truncated)**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NjU…`
- **original-trajectory context**:
  > ....login(username='9654124977', password='qlOZ]NT'))  output: {  "access_token": "**eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NjU0MTI0OTc3IiwiZXhwIjoxNjg0NDEyMDk4fQ.jYAPORCmfNo6R-hEZGM0VLPsOIPdYZ44MzObDIbADSQ**",  "token_type": "Bearer" } ...

## Headline numbers

| metric | value |
|---|---|
| sample preservation count | **6 / 8** |
| pass rate WITH token preserved  | **83%** |
| pass rate WITHOUT token (dropped) | **0%** |
| **pass-rate gap (with − without)** | **+83 pp** |
| greedy preserved this token? | True |
| greedy pass | True |
| full-context baseline | False |
| sample preserve pattern (submission order) | `[1, 0, 1, 0, 1, 1, 1, 1]` |
| sample pass pattern | `[1, 0, 1, 0, 1, 1, 1, 0]` |

## Task instruction

> I am going on a vacation. Move my go-to-sleep phone alarm to 20 minutes later and disable the rest.


## v5 / v7 narrative for this case

The token `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NjU…` is present in the original AppWorld trajectory; under K=2 recompression in the `general_task_agnostic` family, **2 of 8 stochastic samples drop it**, while 6 preserve it. Samples that preserve the token pass downstream at 83%; samples that drop it pass at 0% (+83 pp gap). This is the v5 'recovered-then-dropped' phenomenon and the v7 'form-prior conditions on fact_type, not need_label' phenomenon manifesting at sample-distribution level: the same compressor applied to the same input drops critical functional content with non-trivial probability, and best-of-N selection exploits this variance.


## Greedy compressed — 366 chars (candidate `302c169_3__general_task_agnostic__greedy`)

- token `jwt_token` preserved: **True**
- snippet around token: > ... Login: POST /auth/token, user=9654124977, pass=qlOZ]NT - Success: access_token=**eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NjU0MTI0OTc3IiwiZXhwIjoxNjg0NDEyMDk4fQ.jYAPORCmfNo6R-hEZGM0VLPsOIPdYZ44MzObDIbADSQ** - Warning: use print statements only in action fields to avoid syntax errors...

```
Goal: modify phone alarms
- API: GET /alarms (requires access_token)
- Login: POST /auth/token, user=9654124977, pass=qlOZ]NT
- Success: access_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NjU0MTI0OTc3IiwiZXhwIjoxNjg0NDEyMDk4fQ.jYAPORCmfNo6R-hEZGM0VLPsOIPdYZ44MzObDIbADSQ
- Warning: use print statements only in action fields to avoid syntax errors
```

## Passing sample (token preserved) — 443 chars (candidate `302c169_3__general_task_agnostic__sample_00`)

- token `jwt_token` preserved: **True**
- snippet around token: > ...s: username=9654124977, password=qlOZ]NT  Logged in successfully. Access token: **eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NjU0MTI0OTc3IiwiZXhwIjoxNjg0NDEyMDk4fQ.jYAPORCmfNo6R-hEZGM0VLPsOIPdYZ44MzObDIbADSQ**  Next step: Call show_alarms API with access token....

```
Goal: Modify phone alarms using phone app.

APIs:
- login: POST /auth/token (username=phone_number, password)
- show_alarms: GET /alarms (requires access_token)

Credentials: username=9654124977, password=qlOZ]NT

Logged in successfully. Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs5NjU0MTI0OTc3IiwiZXhwIjoxNjg0NDEyMDk4fQ.jYAPORCmfNo6R-hEZGM0VLPsOIPdYZ44MzObDIbADSQ

Next step: Call show_alarms API with access token.
```

## Failing sample (token dropped, representative) — 327 chars (candidate `302c169_3__general_task_agnostic__sample_02`)

- token `jwt_token` preserved: **False**

```
Goal: Modify phone alarms.

APIs:
- show_alarms: GET /alarms, requires access_token, optional page_index (default 0), page_limit (default 5, max 20)
- Login: POST /auth/token, requires phone_number and password

Credentials: phone=9654124977, password=qlOZ]NT

Current state: Logged in, access token ready for alarm operations.
```