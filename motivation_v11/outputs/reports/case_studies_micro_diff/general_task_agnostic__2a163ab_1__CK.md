# Micro-diff case study — `2a163ab_1` / `general_task_agnostic` / round `CK`

## Headline numbers

| metric | value |
|---|---|
| mean pairwise textual similarity (8 samples) | **0.8424** |
| sample pass count (of 8) | **3 / 8** |
| sample pass pattern (in submission order) | `[0, 1, 1, 0, 0, 1, 0, 0]` |
| char-length range across samples | 291 |
| char-length (passing) − (failing) representative | +188 |
| pair sim (passing rep vs failing rep) | **0.4408** |
| pair sim (passing rep vs greedy) | 0.2740 |
| full-context baseline | False |
| greedy compressed | False |

## Task instruction

> Like all the venmo transactions from today involving any of my roommates on my venmo social feed.


## Why this case is surprising

3 of 8 samples pass the downstream agent, yet pairwise textual similarity averages 0.842 (i.e., 15.8% of the text differs on average). The passing sample is 44.1% similar to a representative failing sample, yet flips the outcome. This is the v11 "micro-perturbation behavioral entropy" claim made concrete.


## Greedy compressed text — 430 chars

```
Task: Find roommates from contacts, get today's venmo social feed, like roommate transactions.

Apps explored:
- supervisor: no contacts API, has show_account_passwords
- phone: show_contact_relationships (requires login)
- venmo: for social feed

Passwords obtained:
- phone: %oOZCyw
- venmo: +vNSutx

Next steps: Login to phone app → get contacts/relationships → identify roommates → get venmo feed → like roommate transactions.
```


## Passing sample — 643 chars (candidate `2a163ab_1__general_task_agnostic__sample_01`)

```
Goal: Find roommates from contacts, get today's venmo social feed, like roommate transactions.

APIs:
- Supervisor: show_active_task, complete_task, show_profile, show_addresses, show_payment_cards, show_account_passwords
- Phone: login (needs username/password), show_contact_relationships
- Other apps: amazon, file_system, spotify, venmo, gmail, splitwise, simple_note, todoist

Credentials:
- phone password: %oOZCyw
- venmo password: +vNSutx

Key constraint: show_contact_relationships requires access_token from phone login first.

Issue to avoid: Syntax errors with comments inside print() - use plain print(apis.xxx.yyy()) format only.
```


## Failing sample (representative) — 455 chars (candidate `2a163ab_1__general_task_agnostic__sample_00`)

```
Goal: Find roommates from contacts, get today's Venmo social feed, like their transactions.

Available apps: phone, venmo, amazon, spotify, file_system
Supervisor functions: show_active_task, complete_task, show_profile, show_addresses, show_payment_cards, show_account_passwords

Credentials: phone: %oOZCyw, venmo: +vNSutx

Plan: Login to phone → get contacts → identify roommates → login to Venmo → get today's social feed → like roommate transactions.
```


## Diff: failing → passing

```diff
--- failing
+++ passing
@@ -1,8 +1,14 @@
-Goal: Find roommates from contacts, get today's Venmo social feed, like their transactions.
+Goal: Find roommates from contacts, get today's venmo social feed, like roommate transactions.
 
-Available apps: phone, venmo, amazon, spotify, file_system
-Supervisor functions: show_active_task, complete_task, show_profile, show_addresses, show_payment_cards, show_account_passwords
+APIs:
+- Supervisor: show_active_task, complete_task, show_profile, show_addresses, show_payment_cards, show_account_passwords
+- Phone: login (needs username/password), show_contact_relationships
+- Other apps: amazon, file_system, spotify, venmo, gmail, splitwise, simple_note, todoist
 
-Credentials: phone: %oOZCyw, venmo: +vNSutx
+Credentials:
+- phone password: %oOZCyw
+- venmo password: +vNSutx
 
-Plan: Login to phone → get contacts → identify roommates → login to Venmo → get today's social feed → like roommate transactions.
+Key constraint: show_contact_relationships requires access_token from phone login first.
+
+Issue to avoid: Syntax errors with comments inside print() - use plain print(apis.xxx.yyy()) format only.
```


## Diff: greedy → passing

```diff
--- greedy
+++ passing
@@ -1,12 +1,14 @@
-Task: Find roommates from contacts, get today's venmo social feed, like roommate transactions.
+Goal: Find roommates from contacts, get today's venmo social feed, like roommate transactions.
 
-Apps explored:
-- supervisor: no contacts API, has show_account_passwords
-- phone: show_contact_relationships (requires login)
-- venmo: for social feed
+APIs:
+- Supervisor: show_active_task, complete_task, show_profile, show_addresses, show_payment_cards, show_account_passwords
+- Phone: login (needs username/password), show_contact_relationships
+- Other apps: amazon, file_system, spotify, venmo, gmail, splitwise, simple_note, todoist
 
-Passwords obtained:
-- phone: %oOZCyw
-- venmo: +vNSutx
+Credentials:
+- phone password: %oOZCyw
+- venmo password: +vNSutx
 
-Next steps: Login to phone app → get contacts/relationships → identify roommates → get venmo feed → like roommate transactions.
+Key constraint: show_contact_relationships requires access_token from phone login first.
+
+Issue to avoid: Syntax errors with comments inside print() - use plain print(apis.xxx.yyy()) format only.
```
