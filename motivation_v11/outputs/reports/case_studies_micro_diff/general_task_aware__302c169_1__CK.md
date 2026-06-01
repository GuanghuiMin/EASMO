# Micro-diff case study — `302c169_1` / `general_task_aware` / round `CK`

## Headline numbers

| metric | value |
|---|---|
| mean pairwise textual similarity (8 samples) | **0.8721** |
| sample pass count (of 8) | **5 / 8** |
| sample pass pattern (in submission order) | `[0, 1, 1, 1, 0, 1, 0, 1]` |
| char-length range across samples | 173 |
| char-length (passing) − (failing) representative | +137 |
| pair sim (passing rep vs failing rep) | **0.3960** |
| pair sim (passing rep vs greedy) | 0.3626 |
| full-context baseline | False |
| greedy compressed | False |

## Task instruction

> I am going on a vacation. Move my go-to-sleep phone alarm to 1 hour later and disable the rest.


## Why this case is surprising

5 of 8 samples pass the downstream agent, yet pairwise textual similarity averages 0.872 (i.e., 12.8% of the text differs on average). The passing sample is 39.6% similar to a representative failing sample, yet flips the outcome. This is the v11 "micro-perturbation behavioral entropy" claim made concrete.


## Greedy compressed text — 617 chars

```
Task: Move go-to-sleep alarm to 1 hour later, disable all other alarms.

Auth:
- Username: 4436271690
- Password: 1YEfc]J
- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs0NDM2MjcxNjkwIiwiZXhwIjoxNjg0NDEyMDk4fQ.sH_fu2BLPemoZ5K7AtlRb-Fl9VojSGs3B3Os0nd9uGo

Completed:
- Alarm 122 ("Go to sleep"): 21:30 → 22:30 ✓
- Alarm 120 ("Wake Up", 07:30 weekdays): DISABLED ✓
- Alarm 121 ("Wake Up", 08:30 weekends): DISABLED ✓
- Alarm 126 ("Pick up Laundry", 17:30): DISABLED ✓

Remaining:
1. Check alarm 124 (10:00, all days) - verify exists and disable if enabled
2. Verify all changes applied correctly
```


## Passing sample — 624 chars (candidate `302c169_1__general_task_aware__sample_02`)

```
Task: Move go-to-sleep phone alarm to 1 hour later and disable all other alarms.

Completed:
- Modified alarm 122 ("Go to sleep"): 21:30 → 22:30 ✓
- Disabled alarm 120 (Wake Up, 07:30 weekdays) ✓
- Disabled alarm 121 (Wake Up, 08:30 weekends) ✓
- Disabled alarm 126 (Pick up Laundry, 17:30) ✓

Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs0NDM2MjcxNjkwIiwiZXhwIjoxNjg0NDEyMDk4fQ.sH_fu2BLPemoZ5K7AtlRb-Fl9VojSGs3B3Os0nd9uGo

API: apis.phone.update_alarm(alarm_id, access_token, time=?, enabled=?)

Remaining: Check alarm 124 (was in truncated list) - verify if it exists and disable if it's an alarm.
```


## Failing sample (representative) — 487 chars (candidate `302c169_1__general_task_aware__sample_00`)

```
- Task: Move "go-to-sleep" alarm 1 hour later, disable all other alarms
- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs0NDM2MjcxNjkwIiwiZXhwIjoxNjg0NDEyMDk4fQ.sH_fu2BLPemoZ5K7AtlRb-Fl9VojSGs3B3Os0nd9uGo
- Completed: alarm_id 122 ("Go to sleep") updated from 21:30 to 22:30
- Completed: disabled alarm_ids 120, 121, 126
- Remaining: Check alarm_id 124 - disable if exists and is not "Go to sleep" alarm
- Final step: Verify all non-go-to-sleep alarms are disabled
```


## Diff: failing → passing

```diff
--- failing
+++ passing
@@ -1,6 +1,13 @@
-- Task: Move "go-to-sleep" alarm 1 hour later, disable all other alarms
-- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs0NDM2MjcxNjkwIiwiZXhwIjoxNjg0NDEyMDk4fQ.sH_fu2BLPemoZ5K7AtlRb-Fl9VojSGs3B3Os0nd9uGo
-- Completed: alarm_id 122 ("Go to sleep") updated from 21:30 to 22:30
-- Completed: disabled alarm_ids 120, 121, 126
-- Remaining: Check alarm_id 124 - disable if exists and is not "Go to sleep" alarm
-- Final step: Verify all non-go-to-sleep alarms are disabled
+Task: Move go-to-sleep phone alarm to 1 hour later and disable all other alarms.
+
+Completed:
+- Modified alarm 122 ("Go to sleep"): 21:30 → 22:30 ✓
+- Disabled alarm 120 (Wake Up, 07:30 weekdays) ✓
+- Disabled alarm 121 (Wake Up, 08:30 weekends) ✓
+- Disabled alarm 126 (Pick up Laundry, 17:30) ✓
+
+Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs0NDM2MjcxNjkwIiwiZXhwIjoxNjg0NDEyMDk4fQ.sH_fu2BLPemoZ5K7AtlRb-Fl9VojSGs3B3Os0nd9uGo
+
+API: apis.phone.update_alarm(alarm_id, access_token, time=?, enabled=?)
+
+Remaining: Check alarm 124 (was in truncated list) - verify if it exists and disable if it's an alarm.
```


## Diff: greedy → passing

```diff
--- greedy
+++ passing
@@ -1,16 +1,13 @@
-Task: Move go-to-sleep alarm to 1 hour later, disable all other alarms.
-
-Auth:
-- Username: 4436271690
-- Password: 1YEfc]J
-- Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs0NDM2MjcxNjkwIiwiZXhwIjoxNjg0NDEyMDk4fQ.sH_fu2BLPemoZ5K7AtlRb-Fl9VojSGs3B3Os0nd9uGo
+Task: Move go-to-sleep phone alarm to 1 hour later and disable all other alarms.
 
 Completed:
-- Alarm 122 ("Go to sleep"): 21:30 → 22:30 ✓
-- Alarm 120 ("Wake Up", 07:30 weekdays): DISABLED ✓
-- Alarm 121 ("Wake Up", 08:30 weekends): DISABLED ✓
-- Alarm 126 ("Pick up Laundry", 17:30): DISABLED ✓
+- Modified alarm 122 ("Go to sleep"): 21:30 → 22:30 ✓
+- Disabled alarm 120 (Wake Up, 07:30 weekdays) ✓
+- Disabled alarm 121 (Wake Up, 08:30 weekends) ✓
+- Disabled alarm 126 (Pick up Laundry, 17:30) ✓
 
-Remaining:
-1. Check alarm 124 (10:00, all days) - verify exists and disable if enabled
-2. Verify all changes applied correctly
+Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs0NDM2MjcxNjkwIiwiZXhwIjoxNjg0NDEyMDk4fQ.sH_fu2BLPemoZ5K7AtlRb-Fl9VojSGs3B3Os0nd9uGo
+
+API: apis.phone.update_alarm(alarm_id, access_token, time=?, enabled=?)
+
+Remaining: Check alarm 124 (was in truncated list) - verify if it exists and disable if it's an alarm.
```
