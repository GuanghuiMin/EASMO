# Micro-diff case study — `6ea6792_3` / `ACON_UTCO` / round `CK`

## Headline numbers

| metric | value |
|---|---|
| mean pairwise textual similarity (8 samples) | **0.9176** |
| sample pass count (of 8) | **7 / 8** |
| sample pass pattern (in submission order) | `[1, 1, 1, 1, 1, 1, 0, 1]` |
| char-length range across samples | 369 |
| char-length (passing) − (failing) representative | -39 |
| pair sim (passing rep vs failing rep) | **0.5549** |
| pair sim (passing rep vs greedy) | 0.4156 |
| full-context baseline | False |
| greedy compressed | True |

## Task instruction

> Accept all pending Venmo payment requests from my coworkers and friends.


## Why this case is surprising

7 of 8 samples pass the downstream agent, yet pairwise textual similarity averages 0.918 (i.e., 8.2% of the text differs on average). The passing sample is 55.5% similar to a representative failing sample, yet flips the outcome. This is the v11 "micro-perturbation behavioral entropy" claim made concrete.


## Greedy compressed text — 1275 chars

```
### REASONING

The agent successfully completed the Venmo task of accepting all pending payment requests. The primary obstacle encountered was authentication - initial API attempts returned 401 errors. The solution was to explicitly pass the access token as a parameter to each API call rather than relying on automatic session handling. The agent retrieved credentials from the supervisor app, logged in, fetched 5 pending payment requests, and approved each one individually using the approve_payment_request endpoint. All pending requests were successfully processed.

### COMPLETED

- Logged into Venmo (paul_mill@gmail.com)
- Retrieved 5 pending payment requests
- Approved all 5 pending requests:
  - Request #3804: $143.00 (Camera Gear) from Jeffrey Smith
  - Request #3462: $44.00 (Fishing License) from Lindsey Simpson
  - Request #5764: Approved
  - Request #3464: Approved
  - Request #4176: Approved

### STATE RETAINED

- **venmo_access_token**: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbytwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.m6K3rMvbzS-jikELGWqWRm9YAe52AmdnqAxNnVFbjVc"
- **approved_payment_request_ids**: [3804, 3462, 5764, 3464, 4176]
- **task_status**: COMPLETED
- **pending_requests_count**: 0 (all requests have been processed)
```


## Passing sample — 1487 chars (candidate `6ea6792_3__ACON_UTCO__sample_01`)

```
### REASONING
The agent successfully completed the task of accepting all pending Venmo payment requests. The workflow involved: (1) exploring available Venmo APIs, (2) encountering a 401 unauthorized error when trying to view payment requests without authentication, (3) retrieving Venmo credentials from the supervisor app, (4) logging in with username `paul_mill@gmail.com` to obtain an access token, (5) re-attempting to view payment requests with the access token, and (6) approving all 5 pending payment requests. The agent learned from the 401 errors that the access token must be explicitly passed to protected endpoints.

### COMPLETED
- **Login to Venmo**: Successfully authenticated with username `paul_mill@gmail.com` and obtained access token
- **Retrieved pending payment requests**: Found 5 pending requests total
- **Approved all 5 payment requests**:
  - Request #3804: $143.00 for "📷Camera Gear" from Jeffrey Smith (jefsmith@gmail.com)
  - Request #3462: $44.00 for "Fishing License" from Lindsey Simpson (lindseysimpson@gmail.com)
  - Request #5764: (approved)
  - Request #3464: (approved)
  - Request #4176: (approved)

### STATE RETAINED
- **already_processed_payment_request_ids**: [3804, 3462, 5764, 3464, 4176]
- **access_token**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbyBwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.m6K3rMvbzS-jikELGWqWRm9YAe52AmdnqAxNnVFbjVc`
- **task_complete**: true — All 5 pending payment requests have been approved.
```


## Failing sample (representative) — 1526 chars (candidate `6ea6792_3__ACON_UTCO__sample_06`)

```
### REASONING
The agent successfully completed the task of accepting all pending Venmo payment requests. The process involved: (1) discovering available Venmo APIs via `show_api_descriptions`, (2) attempting to access payment requests without authentication and receiving a 401 error, (3) retrieving credentials from the supervisor app, (4) logging in to obtain an access token, (5) retrying the payment request fetch with the token, (6) identifying 5 pending requests, and (7) approving each one individually. The key error learned was that the `show_received_payment_requests` API requires an explicit `access_token` parameter even after login—simply being logged in is insufficient.

### COMPLETED
- **Logged into Venmo** with credentials (username: paul_mill@gmail.com, password: S1OmnLn)
- **Retrieved pending payment requests**: Found 5 pending requests (IDs: 3804, 3462, 5764, 3464, 4176)
- **Approved all 5 payment requests**:
  - Request 3804 ($143.00, "📷Camera Gear" from Jeffrey Smith) ✓
  - Request 3462 ($44.00, "Fishing License" from Lindsey Simpson) ✓
  - Request 5764 (approved) ✓
  - Request 3464 (approved) ✓
  - Request 4176 (approved) ✓

### STATE RETAINED
- **access_token**: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbytwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.m6K3rMvbzS-jikELGWqWRm9YAe52AmdnqAxNnVFbjVc"
- **venmo_password**: "S1OmnLn"
- **pending_requests_processed**: [3804, 3462, 5764, 3464, 4176]
- **task_status**: COMPLETED — all pending payment requests have been accepted
```


## Diff: failing → passing

```diff
--- failing
+++ passing
@@ -1,18 +1,17 @@
 ### REASONING
-The agent successfully completed the task of accepting all pending Venmo payment requests. The process involved: (1) discovering available Venmo APIs via `show_api_descriptions`, (2) attempting to access payment requests without authentication and receiving a 401 error, (3) retrieving credentials from the supervisor app, (4) logging in to obtain an access token, (5) retrying the payment request fetch with the token, (6) identifying 5 pending requests, and (7) approving each one individually. The key error learned was that the `show_received_payment_requests` API requires an explicit `access_token` parameter even after login—simply being logged in is insufficient.
+The agent successfully completed the task of accepting all pending Venmo payment requests. The workflow involved: (1) exploring available Venmo APIs, (2) encountering a 401 unauthorized error when trying to view payment requests without authentication, (3) retrieving Venmo credentials from the supervisor app, (4) logging in with username `paul_mill@gmail.com` to obtain an access token, (5) re-attempting to view payment requests with the access token, and (6) approving all 5 pending payment requests. The agent learned from the 401 errors that the access token must be explicitly passed to protected endpoints.
 
 ### COMPLETED
-- **Logged into Venmo** with credentials (username: paul_mill@gmail.com, password: S1OmnLn)
-- **Retrieved pending payment requests**: Found 5 pending requests (IDs: 3804, 3462, 5764, 3464, 4176)
+- **Login to Venmo**: Successfully authenticated with username `paul_mill@gmail.com` and obtained access token
+- **Retrieved pending payment requests**: Found 5 pending requests total
 - **Approved all 5 payment requests**:
-  - Request 3804 ($143.00, "📷Camera Gear" from Jeffrey Smith) ✓
-  - Request 3462 ($44.00, "Fishing License" from Lindsey Simpson) ✓
-  - Request 5764 (approved) ✓
-  - Request 3464 (approved) ✓
-  - Request 4176 (approved) ✓
+  - Request #3804: $143.00 for "📷Camera Gear" from Jeffrey Smith (jefsmith@gmail.com)
+  - Request #3462: $44.00 for "Fishing License" from Lindsey Simpson (lindseysimpson@gmail.com)
+  - Request #5764: (approved)
+  - Request #3464: (approved)
+  - Request #4176: (approved)
 
 ### STATE RETAINED
-- **access_token**: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbytwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.m6K3rMvbzS-jikELGWqWRm9YAe52AmdnqAxNnVFbjVc"
-- **venmo_password**: "S1OmnLn"
-- **pending_requests_processed**: [3804, 3462, 5764, 3464, 4176]
-- **task_status**: COMPLETED — all pending payment requests have been accepted
+- **already_processed_payment_request_ids**: [3804, 3462, 5764, 3464, 4176]
+- **access_token**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbyBwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.m6K3rMvbzS-jikELGWqWRm9YAe52AmdnqAxNnVFbjVc`
+- **task_complete**: true — All 5 pending payment requests have been approved.
```


## Diff: greedy → passing

```diff
--- greedy
+++ passing
@@ -1,21 +1,17 @@
 ### REASONING
-
-The agent successfully completed the Venmo task of accepting all pending payment requests. The primary obstacle encountered was authentication - initial API attempts returned 401 errors. The solution was to explicitly pass the access token as a parameter to each API call rather than relying on automatic session handling. The agent retrieved credentials from the supervisor app, logged in, fetched 5 pending payment requests, and approved each one individually using the approve_payment_request endpoint. All pending requests were successfully processed.
+The agent successfully completed the task of accepting all pending Venmo payment requests. The workflow involved: (1) exploring available Venmo APIs, (2) encountering a 401 unauthorized error when trying to view payment requests without authentication, (3) retrieving Venmo credentials from the supervisor app, (4) logging in with username `paul_mill@gmail.com` to obtain an access token, (5) re-attempting to view payment requests with the access token, and (6) approving all 5 pending payment requests. The agent learned from the 401 errors that the access token must be explicitly passed to protected endpoints.
 
 ### COMPLETED
-
-- Logged into Venmo (paul_mill@gmail.com)
-- Retrieved 5 pending payment requests
-- Approved all 5 pending requests:
-  - Request #3804: $143.00 (Camera Gear) from Jeffrey Smith
-  - Request #3462: $44.00 (Fishing License) from Lindsey Simpson
-  - Request #5764: Approved
-  - Request #3464: Approved
-  - Request #4176: Approved
+- **Login to Venmo**: Successfully authenticated with username `paul_mill@gmail.com` and obtained access token
+- **Retrieved pending payment requests**: Found 5 pending requests total
+- **Approved all 5 payment requests**:
+  - Request #3804: $143.00 for "📷Camera Gear" from Jeffrey Smith (jefsmith@gmail.com)
+  - Request #3462: $44.00 for "Fishing License" from Lindsey Simpson (lindseysimpson@gmail.com)
+  - Request #5764: (approved)
+  - Request #3464: (approved)
+  - Request #4176: (approved)
 
 ### STATE RETAINED
-
-- **venmo_access_token**: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbytwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.m6K3rMvbzS-jikELGWqWRm9YAe52AmdnqAxNnVFbjVc"
-- **approved_payment_request_ids**: [3804, 3462, 5764, 3464, 4176]
-- **task_status**: COMPLETED
-- **pending_requests_count**: 0 (all requests have been processed)
+- **already_processed_payment_request_ids**: [3804, 3462, 5764, 3464, 4176]
+- **access_token**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2ZW5tbyBwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.m6K3rMvbzS-jikELGWqWRm9YAe52AmdnqAxNnVFbjVc`
+- **task_complete**: true — All 5 pending payment requests have been approved.
```
