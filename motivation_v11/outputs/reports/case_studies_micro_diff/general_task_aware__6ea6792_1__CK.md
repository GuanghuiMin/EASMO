# Micro-diff case study — `6ea6792_1` / `general_task_aware` / round `CK`

## Headline numbers

| metric | value |
|---|---|
| mean pairwise textual similarity (8 samples) | **0.8647** |
| sample pass count (of 8) | **3 / 8** |
| sample pass pattern (in submission order) | `[0, 0, 1, 0, 0, 1, 0, 1]` |
| char-length range across samples | 363 |
| char-length (passing) − (failing) representative | +228 |
| pair sim (passing rep vs failing rep) | **0.4513** |
| pair sim (passing rep vs greedy) | 0.3770 |
| full-context baseline | False |
| greedy compressed | False |

## Task instruction

> Accept all pending Venmo payment requests from my roommates and coworkers.


## Why this case is surprising

3 of 8 samples pass the downstream agent, yet pairwise textual similarity averages 0.865 (i.e., 13.5% of the text differs on average). The passing sample is 45.1% similar to a representative failing sample, yet flips the outcome. This is the v11 "micro-perturbation behavioral entropy" claim made concrete.


## Greedy compressed text — 882 chars

```
**Task:** Accept all pending Venmo payment requests from roommates and coworkers.

**Credentials:**
- Email: nan_ritt@gmail.com
- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbytuYW5fcml0dEBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.Hb6XM_sfWFrFryqPNdGauBJxCLXAmiIk9p9Se3_DUB0

**Pending requests:**
- payment_request_id=1896, amount=$41, sender=Chelsea Burch (chelsea-burch@gmail.com), description="Art Supplies", created_at="2023-05-16T02:00:24"
- Request id=3729 ($11) was already denied

**Issue:** Need to identify which pending requests are from roommates/coworkers. Attempted show_contact_relationships but received 401 error (likely auth issue).

**Approach needed:**
1. Find API to accept/approve payment request
2. Determine how to identify roommates/coworkers (alternative to show_contact_relationships)
3. Accept each pending request from valid contacts
```


## Passing sample — 1171 chars (candidate `6ea6792_1__general_task_aware__sample_01`)

```
**Task:** Accept all pending Venmo payment requests from roommates and coworkers.

**Venmo Credentials:**
- Email: nan_ritt@gmail.com / Password: OzVS[j5
- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbytuYW5fcml0dEBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.Hb6XM_sfWFrFryqPNdGauBJxCLXAmiIk9p9Se3_DUB0

**Pending Request Found:**
- ID 1896: $41 from Chelsea Burch (chelsea-burch@gmail.com) - PENDING

**Previously Denied (do not repeat):**
- ID 3729: $11 from Marcus Smith (ma_smith@gmail.com) - DENIED

**Failed Attempts (avoid):**
- `show_received_payment_requests` without access_token → 401 error
- `show_contact_relationships` without access_token → 401 error

**Phone App (for contact info):**
- Username: 2307354647, Password: !5VAxgi
- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSsyMzA3MzU0NjQ3IiwiZXhwIjoxNjg0NDEyNzQxfQ.b-sDdvFOafKBAOniy9yN3fKopwG0RAi8nXABAfEcCtE

**Next Steps:**
1. Call `show_received_payment_requests(access_token=...)` to get full list
2. Use `show_contact_relationships(access_token=...)` to identify roommates vs coworkers
3. Accept each pending request using appropriate API (not yet identified)
```


## Failing sample (representative) — 943 chars (candidate `6ea6792_1__general_task_aware__sample_00`)

```
**Task:** Accept all pending Venmo payment requests from roommates and coworkers.

**Credentials:**
- Username: nan_ritt@gmail.com
- Password: OzVS[j5
- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbytuYW5fcml0dEBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.Hb6XM_sfWFrFryqPNdGauBJxCLXAmiIk9p9Se3_DUB0

**Pending request found:**
- payment_request_id: 1896
- Amount: $41.00
- Sender: Chelsea Burch (chelsea-burch@gmail.com)
- Description: "🎨Art Supplies"

**Failed attempts (do not repeat):**
- show_received_payment_requests failed with 401 - missing access_token parameter
- show_contact_relationships failed with 401 - missing access_token parameter

**Next steps:**
1. Use access_token in API calls (required to avoid 401 errors)
2. Identify roommates/coworkers (check phone contacts or Venmo contacts)
3. Find the correct API to accept/approve payment requests
4. Accept pending requests from identified roommates/coworkers
```


## Diff: failing → passing

```diff
--- failing
+++ passing
@@ -1,22 +1,24 @@
 **Task:** Accept all pending Venmo payment requests from roommates and coworkers.
 
-**Credentials:**
-- Username: nan_ritt@gmail.com
-- Password: OzVS[j5
+**Venmo Credentials:**
+- Email: nan_ritt@gmail.com / Password: OzVS[j5
 - Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbytuYW5fcml0dEBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.Hb6XM_sfWFrFryqPNdGauBJxCLXAmiIk9p9Se3_DUB0
 
-**Pending request found:**
-- payment_request_id: 1896
-- Amount: $41.00
-- Sender: Chelsea Burch (chelsea-burch@gmail.com)
-- Description: "🎨Art Supplies"
+**Pending Request Found:**
+- ID 1896: $41 from Chelsea Burch (chelsea-burch@gmail.com) - PENDING
 
-**Failed attempts (do not repeat):**
-- show_received_payment_requests failed with 401 - missing access_token parameter
-- show_contact_relationships failed with 401 - missing access_token parameter
+**Previously Denied (do not repeat):**
+- ID 3729: $11 from Marcus Smith (ma_smith@gmail.com) - DENIED
 
-**Next steps:**
-1. Use access_token in API calls (required to avoid 401 errors)
-2. Identify roommates/coworkers (check phone contacts or Venmo contacts)
-3. Find the correct API to accept/approve payment requests
-4. Accept pending requests from identified roommates/coworkers
+**Failed Attempts (avoid):**
+- `show_received_payment_requests` without access_token → 401 error
+- `show_contact_relationships` without access_token → 401 error
+
+**Phone App (for contact info):**
+- Username: 2307354647, Password: !5VAxgi
+- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSsyMzA3MzU0NjQ3IiwiZXhwIjoxNjg0NDEyNzQxfQ.b-sDdvFOafKBAOniy9yN3fKopwG0RAi8nXABAfEcCtE
+
+**Next Steps:**
+1. Call `show_received_payment_requests(access_token=...)` to get full list
+2. Use `show_contact_relationships(access_token=...)` to identify roommates vs coworkers
+3. Accept each pending request using appropriate API (not yet identified)
```


## Diff: greedy → passing

```diff
--- greedy
+++ passing
@@ -1,16 +1,24 @@
 **Task:** Accept all pending Venmo payment requests from roommates and coworkers.
 
-**Credentials:**
-- Email: nan_ritt@gmail.com
+**Venmo Credentials:**
+- Email: nan_ritt@gmail.com / Password: OzVS[j5
 - Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbytuYW5fcml0dEBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.Hb6XM_sfWFrFryqPNdGauBJxCLXAmiIk9p9Se3_DUB0
 
-**Pending requests:**
-- payment_request_id=1896, amount=$41, sender=Chelsea Burch (chelsea-burch@gmail.com), description="Art Supplies", created_at="2023-05-16T02:00:24"
-- Request id=3729 ($11) was already denied
+**Pending Request Found:**
+- ID 1896: $41 from Chelsea Burch (chelsea-burch@gmail.com) - PENDING
 
-**Issue:** Need to identify which pending requests are from roommates/coworkers. Attempted show_contact_relationships but received 401 error (likely auth issue).
+**Previously Denied (do not repeat):**
+- ID 3729: $11 from Marcus Smith (ma_smith@gmail.com) - DENIED
 
-**Approach needed:**
-1. Find API to accept/approve payment request
-2. Determine how to identify roommates/coworkers (alternative to show_contact_relationships)
-3. Accept each pending request from valid contacts
+**Failed Attempts (avoid):**
+- `show_received_payment_requests` without access_token → 401 error
+- `show_contact_relationships` without access_token → 401 error
+
+**Phone App (for contact info):**
+- Username: 2307354647, Password: !5VAxgi
+- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSsyMzA3MzU0NjQ3IiwiZXhwIjoxNjg0NDEyNzQxfQ.b-sDdvFOafKBAOniy9yN3fKopwG0RAi8nXABAfEcCtE
+
+**Next Steps:**
+1. Call `show_received_payment_requests(access_token=...)` to get full list
+2. Use `show_contact_relationships(access_token=...)` to identify roommates vs coworkers
+3. Accept each pending request using appropriate API (not yet identified)
```
