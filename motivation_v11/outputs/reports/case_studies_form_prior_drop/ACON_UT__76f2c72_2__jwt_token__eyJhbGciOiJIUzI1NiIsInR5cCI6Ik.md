# Form-prior drop case — `76f2c72_2` / `ACON_UT` / round `CK`

## Token under analysis

- **category**: `jwt_token`
- **value (truncated)**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3R…`
- **original-trajectory context**:
  > ...me='chelsea-burch@gmail.com', password='oD)7^7D'))  output: {  "access_token": "**eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStjaGVsc2VhLWJ1cmNoQGdtYWlsLmNvbSIsImV4cCI6MTY4NDQxMjA5OH0.nSvPoeEUdfNdsM4pBcFdZsFYS-QBhBNL2-9jBNaJPp8**",  "token_type": "Bearer" }  ### step 7 action: # Now let me explore the bills ...

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
| sample preserve pattern (submission order) | `[1, 0, 0, 1, 1, 1, 1, 1]` |
| sample pass pattern | `[1, 0, 0, 0, 1, 1, 1, 1]` |

## Task instruction

> What is the total cost of my internet bills for this year? The bills are in "~/bills/" directory of my file system.


## v5 / v7 narrative for this case

The token `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3R…` is present in the original AppWorld trajectory; under K=2 recompression in the `ACON_UT` family, **2 of 8 stochastic samples drop it**, while 6 preserve it. Samples that preserve the token pass downstream at 83%; samples that drop it pass at 0% (+83 pp gap). This is the v5 'recovered-then-dropped' phenomenon and the v7 'form-prior conditions on fact_type, not need_label' phenomenon manifesting at sample-distribution level: the same compressor applied to the same input drops critical functional content with non-trivial probability, and best-of-N selection exploits this variance.


## Greedy compressed — 1198 chars (candidate `76f2c72_2__ACON_UT__greedy`)

- token `jwt_token` preserved: **True**
- snippet around token: > ... - Subtotal calculated: $314 (Jan-Apr 2023)  ### DATA RETAINED - Access token: `**eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStjaGVsc2VhLWJ1cmNoQGdtYWlsLmNvbSIsImV4cCI6MTY4NDQxMjA5OH0.nSvPoeEUdfNdsM4pBcFdZsFYS-QBhBNL2-9jBNaJPp8**` - Internet bills path: `/home/chelsea/bills/internet/` - Processed files: 2023...

```
### REASONING
The agent is calculating the total cost of internet bills for 2023. Four monthly bills have been retrieved (Jan-Apr) with a combined subtotal of $314. The agent needs to continue checking for remaining 2023 months (starting with 2023-05.txt) to complete the task. The access token and directory path are retained for resuming the file listing operation.

### COMPLETED
- Logged into file_system app with credentials (chelsea-burch@gmail.com)
- Listed contents of ~/bills/internet/ directory
- Retrieved and processed 4 internet bill files for 2023:
  - 2023-01.txt: $72
  - 2023-02.txt: $79
  - 2023-03.txt: $84
  - 2023-04.txt: $79
- Current year confirmed as 2023
- Subtotal calculated: $314 (Jan-Apr 2023)

### DATA RETAINED
- Access token: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStjaGVsc2VhLWJ1cmNoQGdtYWlsLmNvbSIsImV4cCI6MTY4NDQxMjA5OH0.nSvPoeEUdfNdsM4pBcFdZsFYS-QBhBNL2-9jBNaJPp8`
- Internet bills path: `/home/chelsea/bills/internet/`
- Processed files: 2023-01.txt through 2023-04.txt
- Subtotal (Jan-Apr 2023): $314
- Remaining task: Check for 2023-05.txt through 2023-12.txt, retrieve any additional 2023 bills, and calculate final total for the year
```

## Passing sample (token preserved) — 1433 chars (candidate `76f2c72_2__ACON_UT__sample_00`)

- token `jwt_token` preserved: **True**
- snippet around token: > ...- 2023-04.txt: $79 - Current running total: $314  ### ARTIFACTS - Access token: **eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStjaGVsc2VhLWJ1cmNoQGdtYWlsLmNvbSIsImV4cCI6MTY4NDQxMjA5OH0.nSvPoeEUdfNdsM4pBcFdZsFYS-QBhBNL2-9jBNaJPp8** - Internet bills path: /home/chelsea/bills/internet/ - Bill files for 2023: 202...

```
### REASONING
The agent is calculating the total cost of 2023 internet bills from ~/bills/internet/. After exploring available APIs, the agent identified file_system as the relevant app, retrieved credentials from the supervisor app, and successfully obtained an access token. The agent discovered the bills directory contains subdirectories for cable, electricity, and internet. Four internet bill files (January through April 2023) have been read so far, totaling $314. The agent verified the current year is 2023 and identified that 2023-05.txt exists but hasn't been read yet. The agent should now read the May bill to complete the calculation.

### COMPLETED
- Explored available APIs and identified file_system as the relevant app
- Retrieved credentials from supervisor app and obtained access token
- Discovered bills directory structure: ~/bills/ contains cable/, electricity/, and internet/ subdirectories
- Retrieved and parsed internet bills for 2023:
  - 2023-01.txt: $72
  - 2023-02.txt: $79
  - 2023-03.txt: $84
  - 2023-04.txt: $79
- Current running total: $314

### ARTIFACTS
- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStjaGVsc2VhLWJ1cmNoQGdtYWlsLmNvbSIsImV4cCI6MTY4NDQxMjA5OH0.nSvPoeEUdfNdsM4pBcFdZsFYS-QBhBNL2-9jBNaJPp8
- Internet bills path: /home/chelsea/bills/internet/
- Bill files for 2023: 2023-01.txt through 2023-05.txt (2023-05.txt needs to be read)
- Current year: 2023
```

## Failing sample (token dropped, representative) — 925 chars (candidate `76f2c72_2__ACON_UT__sample_03`)

- token `jwt_token` preserved: **False**

```
## REASONING
The agent is calculating the total cost of 2023 internet bills. Progress so far:
- Successfully authenticated to file_system using supervisor credentials (token retained for operations)
- Discovered bills at /home/chelsea/bills/internet/ with YYYY-MM.txt naming pattern
- Retrieved 4 bills for early 2023 (Jan-Apr)
- Current year confirmed as 2023
- The access token remains valid and is retained for continued file operations
- Agent needs to retrieve the remaining 8 months (May-December) to complete the calculation

## COMPLETED
- Authenticated to file_system app (token retained for operations)
- Located internet bills at /home/chelsea/bills/internet/
- Retrieved 4 bills for 2023:
  - 2023-01: $72
  - 2023-02: $79
  - 2023-03: $84
  - 2023-04: $79
- Running total so far: $314

## PENDING
- Read remaining 2023 bills (2023-05 through 2023-12)
- Calculate final sum of all 12 months
- Report total to user
```