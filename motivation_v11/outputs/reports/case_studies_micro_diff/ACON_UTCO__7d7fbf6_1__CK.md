# Micro-diff case study — `7d7fbf6_1` / `ACON_UTCO` / round `CK`

## Headline numbers

| metric | value |
|---|---|
| mean pairwise textual similarity (8 samples) | **0.9292** |
| sample pass count (of 8) | **5 / 8** |
| sample pass pattern (in submission order) | `[1, 1, 0, 1, 0, 1, 0, 1]` |
| char-length range across samples | 299 |
| char-length (passing) − (failing) representative | -150 |
| pair sim (passing rep vs failing rep) | **0.4267** |
| pair sim (passing rep vs greedy) | 0.3041 |
| full-context baseline | False |
| greedy compressed | True |

## Task instruction

> The "~/photographs/" directory in my file system has photo files organized in sub-directories for each vacation spot. Compress them and save them in "~/photographs/vacations/<vacation_spot>.zip" for each vacation spot, and then delete all vacation spot sub-directories. Here, <vacation_spot> is the name of the vacation spot as it appears in the sub-directory name.


## Why this case is surprising

5 of 8 samples pass the downstream agent, yet pairwise textual similarity averages 0.929 (i.e., 7.1% of the text differs on average). The passing sample is 42.7% similar to a representative failing sample, yet flips the outcome. This is the v11 "micro-perturbation behavioral entropy" claim made concrete.


## Greedy compressed text — 1840 chars

```
### REASONING
The agent is working on compressing vacation photos from sub-directories in ~/photographs/. Progress so far:
- Successfully authenticated to file_system app and obtained access_token
- Encountered 401 errors when calling show_directory - identified root causes: (1) used wrong parameter name "path" instead of "directory_path", (2) did not pass access_token to authenticated endpoints
- Reviewed API docs and confirmed correct parameters: access_token (required), directory_path (optional, defaults to "/")
- Next step: call show_directory with correct parameters to discover vacation spot sub-directories before proceeding with compression

### COMPLETED
- [x] Retrieved file_system credentials from supervisor app
- [x] Successfully logged in to file_system app and obtained access_token
- [x] Identified correct API parameters for show_directory (access_token required, directory_path not "path")
- [x] Cached API documentation for login and show_directory endpoints

### STATE RETAINED
- **file_system_access_token**: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStuaWNob2xhcy53ZWJlckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.ExTO-mb9d0PRD0cwHmzRMrkZ5UhYEXtlaLQAr4yiScE"
- **target_directory**: "~/photographs/"
- **output_directory**: "~/photographs/vacations/"
- **cached_api_results**: login API documentation, show_directory API documentation (both retrieved)
- **error_context**: 401 errors occurred because: (1) used wrong parameter name "path" instead of "directory_path", (2) did not pass access_token to authenticated endpoints
- **processing_progress**: Not started - need to discover vacation spot sub-directories first
- **vacation_spots_to_process**: Not yet discovered (pending show_directory call)
- **api_call_correct_params**: {"access_token": "<token>", "directory_path": "~/photographs/"}
```


## Passing sample — 1863 chars (candidate `7d7fbf6_1__ACON_UTCO__sample_01`)

```
### REASONING
The agent is working on compressing photo files organized in vacation spot sub-directories within ~/photographs/. The primary challenge encountered was authentication - the file_system API requires an access_token parameter for protected endpoints. The agent successfully obtained credentials from the supervisor app, logged in, and retrieved an access token. However, the agent has not yet successfully called show_directory with the token to list the vacation spot sub-directories. The next step is to use the obtained access token to list the contents of ~/photographs/ to identify all vacation spot sub-directories, then compress each to .zip format.

### COMPLETED
- Retrieved file_system credentials from supervisor app (username: nicholas.weber@gmail.com, password: "Ia$)7$5")
- Successfully logged into file_system app and obtained access_token
- Reviewed API documentation for show_directory (requires access_token parameter)

### STATE RETAINED
- **file_system_credentials**: username=nicholas.weber@gmail.com, password=Ia$)7$5
- **file_system_access_token**: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStuaWNob2xhcy53ZWJlckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.ExTO-mb9d0PRD0cwHmzRMrkZ5UhYEXtlaLQAr4yiScE
- **target_directory**: ~/photographs/
- **destination_directory**: ~/photographs/vacations/
- **cached_api_results**: show_directory API requires access_token as required parameter, directory_path as optional parameter with default "/"
- **already_processed_vacation_spots**: [] (none yet - need to discover sub-directories first)
- **processing_progress**: Not yet started - need to call show_directory with access_token to discover sub-directories
- **task_remaining**: List vacation spot sub-directories in ~/photographs/ → compress each to ~/photographs/vacations/<vacation_spot>.zip → delete sub-directories
```


## Failing sample (representative) — 2013 chars (candidate `7d7fbf6_1__ACON_UTCO__sample_02`)

```
### REASONING
The agent is working on compressing photo files organized by vacation spots in ~/photographs/. The main challenge was that the file_system API requires an access_token parameter for authenticated endpoints. The agent successfully obtained credentials from the supervisor app and logged in, receiving an access token. However, subsequent API calls failed with 401 errors because the access_token was not being passed as a parameter to the API calls. The agent discovered through API documentation that show_directory requires an access_token parameter. The next step is to call show_directory with the access_token obtained from the login response to list the contents of ~/photographs/ and identify vacation spot sub-directories.

### COMPLETED
- Obtained file_system credentials from supervisor: username=nicholas.weber@gmail.com, password=Ia$)7$5
- Successfully logged in to file_system app and obtained access_token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStuaWNob2xhcy53ZWJlckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.ExTO-mb9d0PRD0cwHmzRMrkZ5UhYEXtlaLQAr4yiScE
- Discovered API documentation for show_directory - requires access_token and directory_path parameters

### STATE RETAINED
- **file_system_credentials**: username='nicholas.weber@gmail.com', password='Ia$)7$5'
- **access_token**: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStuaWNob2xhcy53ZWJlckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.ExTO-mb9d0PRD0cwHmzRMrkZ5UhYEXtlaLQAr4yiScE
- **target_directory**: ~/photographs/
- **output_directory**: ~/photographs/vacations/
- **vacation_subdirectories**: [] (to be populated)
- **processed_vacation_spots**: [] (list of vacation spots already compressed)
- **task_remaining**:
  1. Call show_directory with access_token to list contents of ~/photographs/ and identify vacation spot sub-directories
  2. For each sub-directory: create zip file at ~/photographs/vacations/<vacation_spot>.zip
  3. Delete each vacation spot sub-directory after compression
```


## Diff: failing → passing

```diff
--- failing
+++ passing
@@ -1,19 +1,17 @@
 ### REASONING
-The agent is working on compressing photo files organized by vacation spots in ~/photographs/. The main challenge was that the file_system API requires an access_token parameter for authenticated endpoints. The agent successfully obtained credentials from the supervisor app and logged in, receiving an access token. However, subsequent API calls failed with 401 errors because the access_token was not being passed as a parameter to the API calls. The agent discovered through API documentation that show_directory requires an access_token parameter. The next step is to call show_directory with the access_token obtained from the login response to list the contents of ~/photographs/ and identify vacation spot sub-directories.
+The agent is working on compressing photo files organized in vacation spot sub-directories within ~/photographs/. The primary challenge encountered was authentication - the file_system API requires an access_token parameter for protected endpoints. The agent successfully obtained credentials from the supervisor app, logged in, and retrieved an access token. However, the agent has not yet successfully called show_directory with the token to list the vacation spot sub-directories. The next step is to use the obtained access token to list the contents of ~/photographs/ to identify all vacation spot sub-directories, then compress each to .zip format.
 
 ### COMPLETED
-- Obtained file_system credentials from supervisor: username=nicholas.weber@gmail.com, password=Ia$)7$5
-- Successfully logged in to file_system app and obtained access_token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStuaWNob2xhcy53ZWJlckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.ExTO-mb9d0PRD0cwHmzRMrkZ5UhYEXtlaLQAr4yiScE
-- Discovered API documentation for show_directory - requires access_token and directory_path parameters
+- Retrieved file_system credentials from supervisor app (username: nicholas.weber@gmail.com, password: "Ia$)7$5")
+- Successfully logged into file_system app and obtained access_token
+- Reviewed API documentation for show_directory (requires access_token parameter)
 
 ### STATE RETAINED
-- **file_system_credentials**: username='nicholas.weber@gmail.com', password='Ia$)7$5'
-- **access_token**: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStuaWNob2xhcy53ZWJlckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.ExTO-mb9d0PRD0cwHmzRMrkZ5UhYEXtlaLQAr4yiScE
+- **file_system_credentials**: username=nicholas.weber@gmail.com, password=Ia$)7$5
+- **file_system_access_token**: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStuaWNob2xhcy53ZWJlckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.ExTO-mb9d0PRD0cwHmzRMrkZ5UhYEXtlaLQAr4yiScE
 - **target_directory**: ~/photographs/
-- **output_directory**: ~/photographs/vacations/
-- **vacation_subdirectories**: [] (to be populated)
-- **processed_vacation_spots**: [] (list of vacation spots already compressed)
-- **task_remaining**:
-  1. Call show_directory with access_token to list contents of ~/photographs/ and identify vacation spot sub-directories
-  2. For each sub-directory: create zip file at ~/photographs/vacations/<vacation_spot>.zip
-  3. Delete each vacation spot sub-directory after compression
+- **destination_directory**: ~/photographs/vacations/
+- **cached_api_results**: show_directory API requires access_token as required parameter, directory_path as optional parameter with default "/"
+- **already_processed_vacation_spots**: [] (none yet - need to discover sub-directories first)
+- **processing_progress**: Not yet started - need to call show_directory with access_token to discover sub-directories
+- **task_remaining**: List vacation spot sub-directories in ~/photographs/ → compress each to ~/photographs/vacations/<vacation_spot>.zip → delete sub-directories
```


## Diff: greedy → passing

```diff
--- greedy
+++ passing
@@ -1,22 +1,17 @@
 ### REASONING
-The agent is working on compressing vacation photos from sub-directories in ~/photographs/. Progress so far:
-- Successfully authenticated to file_system app and obtained access_token
-- Encountered 401 errors when calling show_directory - identified root causes: (1) used wrong parameter name "path" instead of "directory_path", (2) did not pass access_token to authenticated endpoints
-- Reviewed API docs and confirmed correct parameters: access_token (required), directory_path (optional, defaults to "/")
-- Next step: call show_directory with correct parameters to discover vacation spot sub-directories before proceeding with compression
+The agent is working on compressing photo files organized in vacation spot sub-directories within ~/photographs/. The primary challenge encountered was authentication - the file_system API requires an access_token parameter for protected endpoints. The agent successfully obtained credentials from the supervisor app, logged in, and retrieved an access token. However, the agent has not yet successfully called show_directory with the token to list the vacation spot sub-directories. The next step is to use the obtained access token to list the contents of ~/photographs/ to identify all vacation spot sub-directories, then compress each to .zip format.
 
 ### COMPLETED
-- [x] Retrieved file_system credentials from supervisor app
-- [x] Successfully logged in to file_system app and obtained access_token
-- [x] Identified correct API parameters for show_directory (access_token required, directory_path not "path")
-- [x] Cached API documentation for login and show_directory endpoints
+- Retrieved file_system credentials from supervisor app (username: nicholas.weber@gmail.com, password: "Ia$)7$5")
+- Successfully logged into file_system app and obtained access_token
+- Reviewed API documentation for show_directory (requires access_token parameter)
 
 ### STATE RETAINED
-- **file_system_access_token**: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStuaWNob2xhcy53ZWJlckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.ExTO-mb9d0PRD0cwHmzRMrkZ5UhYEXtlaLQAr4yiScE"
-- **target_directory**: "~/photographs/"
-- **output_directory**: "~/photographs/vacations/"
-- **cached_api_results**: login API documentation, show_directory API documentation (both retrieved)
-- **error_context**: 401 errors occurred because: (1) used wrong parameter name "path" instead of "directory_path", (2) did not pass access_token to authenticated endpoints
-- **processing_progress**: Not started - need to discover vacation spot sub-directories first
-- **vacation_spots_to_process**: Not yet discovered (pending show_directory call)
-- **api_call_correct_params**: {"access_token": "<token>", "directory_path": "~/photographs/"}
+- **file_system_credentials**: username=nicholas.weber@gmail.com, password=Ia$)7$5
+- **file_system_access_token**: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStuaWNob2xhcy53ZWJlckBnbWFpbC5jb20iLCJleHAiOjE2ODQ0MTIwOTh9.ExTO-mb9d0PRD0cwHmzRMrkZ5UhYEXtlaLQAr4yiScE
+- **target_directory**: ~/photographs/
+- **destination_directory**: ~/photographs/vacations/
+- **cached_api_results**: show_directory API requires access_token as required parameter, directory_path as optional parameter with default "/"
+- **already_processed_vacation_spots**: [] (none yet - need to discover sub-directories first)
+- **processing_progress**: Not yet started - need to call show_directory with access_token to discover sub-directories
+- **task_remaining**: List vacation spot sub-directories in ~/photographs/ → compress each to ~/photographs/vacations/<vacation_spot>.zip → delete sub-directories
```
