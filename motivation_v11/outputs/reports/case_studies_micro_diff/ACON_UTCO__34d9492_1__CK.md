# Micro-diff case study — `34d9492_1` / `ACON_UTCO` / round `CK`

## Headline numbers

| metric | value |
|---|---|
| mean pairwise textual similarity (8 samples) | **0.9202** |
| sample pass count (of 8) | **1 / 8** |
| sample pass pattern (in submission order) | `[1, 0, 0, 0, 0, 0, 0, 0]` |
| char-length range across samples | 428 |
| char-length (passing) − (failing) representative | +428 |
| pair sim (passing rep vs failing rep) | **0.4277** |
| pair sim (passing rep vs greedy) | 0.3933 |
| full-context baseline | False |
| greedy compressed | False |

## Task instruction

> Arrange my "~/photographs/vacations/" directory by organizing the photos from three vacations. The files created in February and March of this year correspond to Petra and Budapest, respectively, while the others are from Amsterdam. Move them into sub-directories named after their respective vacation spots, maintaining the original file names.


## Why this case is surprising

1 of 8 samples pass the downstream agent, yet pairwise textual similarity averages 0.920 (i.e., 8.0% of the text differs on average). The passing sample is 42.8% similar to a representative failing sample, yet flips the outcome. This is the v11 "micro-perturbation behavioral entropy" claim made concrete.


## Greedy compressed text — 2034 chars

```
### REASONING
The agent has completed all preparatory work: file categorization, destination directory creation, and identification of correct API parameters. The previous attempt to move files failed due to incorrect parameter names, but this has been resolved. The agent now has all information needed to execute the file moves. All 33 files remain to be moved to their respective vacation directories.

### COMPLETED
- Logged into file_system app
- Retrieved complete list of files with creation dates from ~/photographs/vacations/
- Categorized all 33 files by vacation destination based on date patterns
- Created destination directories: Petra, Budapest, Amsterdam
- Identified correct API parameters for move_file (source_file_path, destination_file_path)

### STATE RETAINED
- **access_token**: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.GVP7hSYFgvOWp3S32JRgHhu3OlwGNY7_xn0KbyNxmwY'
- **base_path**: '/home/paul/photographs/vacations/'
- **destination_directories**: ['Petra', 'Budapest', 'Amsterdam']
- **file_categorization**:
  - **Petra** (February 2023): img_02237d7.jpg, img_18b06cc.png, img_1a2b16d.jpeg, img_254fbd9.jpg, img_4d1aafc.jpeg, img_4d7f44a.jpeg, img_5c2f56b.jpeg, img_62679d2.jpeg, img_6d078f8.jpg, img_7e0a7b9.jpg, img_8f2c4d5.jpg, img_9a1b2c3.jpg
  - **Budapest** (March 2023): img_0e15e48.jpg, img_16b39ad.png, img_26cb65d.png, img_3b4c5d6.jpg, img_5d6e7f8.jpg, img_7f8a9b0.jpg, img_9c0d1e2.jpg, img_ad3e4f5.jpg, img_be5f6a7.jpg
  - **Amsterdam** (2022): img_0d36b21.jpeg, img_164a3fe.jpeg, img_29f19c8.jpg, img_2dbefae.png, img_3c4d5e6.jpg, img_4e5f6a7.jpg, img_5f6a7b8.jpg, img_6a7b8c9.jpg, img_7b8c9d0.jpg, img_8c9d0e1.jpg, img_9d0e1f2.jpg, img_ae1f2g3.jpg
- **move_file_parameters**: source_file_path, destination_file_path
- **files_to_move**: All 33 files from file_categorization need to be moved using:
  - source_file_path = base_path + filename
  - destination_file_path = base_path + destination_folder + '/' + filename
```


## Passing sample — 2543 chars (candidate `34d9492_1__ACON_UTCO__sample_00`)

```
### REASONING
The agent is organizing photos from ~/photographs/vacations/ into three subdirectories (Petra, Budapest, Amsterdam) based on file creation dates. After successfully authenticating, identifying the date pattern, and creating the three destination directories, the agent attempted to move files but used incorrect parameter names (`source_path`/`destination_path` instead of `source_file_path`/`destination_file_path`), resulting in a validation error. The agent checked the correct API documentation and now knows the proper parameter names. All 36 files (8 Petra + 8 Budapest + 20 Amsterdam) need to be moved to their respective directories using the correct parameter names.

### COMPLETED
- Logged into file_system app and obtained access token
- Listed all files in ~/photographs/vacations/ directory
- Retrieved creation dates for all files and identified the categorization pattern:
  - February 2023 files (8 files) → Petra
  - March 2023 files (8 files) → Budapest
  - 2022 files (20 files) → Amsterdam
- Created three destination directories: Petra, Budapest, Amsterdam
- Identified correct API parameter names: `source_file_path` and `destination_file_path`

### STATE RETAINED
- **access_token**: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.GVP7hSYFgvOWp3S32JRgHhu3OlwGNY7_xn0KbyNxmwY'
- **file_categorization**:
  - **Petra (February 2023)**: img_02237d7.jpg, img_18b06cc.png, img_1a2b16d.jpeg, img_254fbd9.jpg, img_4d1aafc.jpeg, img_4d7f44a.jpeg, img_5c2f56b.jpeg, img_62679d2.jpeg
  - **Budapest (March 2023)**: img_0e15e48.jpg, img_16b39ad.png, img_26cb65d.png, img_2ecf83d.jpeg, img_301305f.jpg, img_32dc30d.jpeg, img_44361f7.jpeg, img_4e0a1f8.jpg
  - **Amsterdam (2022)**: img_0d36b21.jpeg, img_164a3fe.jpeg, img_29f19c8.jpg, img_2dbefae.png, img_2f4e2a0.png, img_359a9e8.jpeg, img_3c4d9f1.png, img_3f7e5a2.jpeg, img_492c8d3.jpg, img_4a6b8c5.png, img_4f9d2e1.jpeg, img_5371a6b.png, img_56e8c3d.jpeg, img_5a1d9f4.png, img_5d4e7b2.jpeg, img_602c5e8.png, img_63f1a9d.jpeg, img_67d4b2e.png, img_6a8f1c3.jpeg, img_6e2d5b4.png
- **destination_directories**: 
  - ~/photographs/vacations/Petra
  - ~/photographs/vacations/Budapest
  - ~/photographs/vacations/Amsterdam
- **move_file_API**: Use `source_file_path` and `destination_file_path` parameters (not source_path/destination_path)
- **files_to_move**: 36 total (8 Petra + 8 Budapest + 20 Amsterdam)
- **files_moved**: 0 (none moved yet - previous attempt failed due to wrong parameter names)
```


## Failing sample (representative) — 2115 chars (candidate `34d9492_1__ACON_UTCO__sample_01`)

```
### REASONING
The agent has completed photo categorization and directory creation. The only remaining task is moving the 36 files to their respective subdirectories. The previous file move attempt failed because the agent used incorrect parameter names (`source_path`/`destination_path`). The correct parameters are now known: `source_file_path` and `destination_file_path`. All file lists and the destination structure are already identified and stored. The agent can now proceed with moving files using the correct API.

### COMPLETED
- Logged into file_system app and obtained access token
- Listed all 36 files in ~/photographs/vacations/
- Retrieved creation dates for all files to determine categorization
- Created three destination directories: Petra, Budapest, Amsterdam
- Identified correct API parameters for file move operation

### STATE RETAINED
- **access_token**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.GVP7hSYFgvOWp3S32JRgHhu3OlwGNY7_xn0KbyNxmwY`
- **petra_files (Feb 2023)**: ["img_02237d7.jpg", "img_18b06cc.png", "img_1a2b16d.jpeg", "img_254fbd9.jpg", "img_4d1aafc.jpeg", "img_4d7f44a.jpeg", "img_5c2f56b.jpeg", "img_62679d2.jpeg", "img_6d078f8.jpg"]
- **budapest_files (Mar 2023)**: ["img_0e15e48.jpg", "img_16b39ad.png", "img_26cb65d.png", "img_3a2b1c9.jpg", "img_44361f7.jpeg", "img_4734c9e.png", "img_4e5f6a7.jpeg", "img_5248b3e.jpg", "img_5d6c7e8.jpeg", "img_6159d8f.jpg", "img_68a7e9f.jpeg", "img_7532f1a.png"]
- **amsterdam_files (2022)**: ["img_0d36b21.jpeg", "img_164a3fe.jpeg", "img_29f19c8.jpg", "img_2dbefae.png", "img_301305f.jpg", "img_32dc30d.jpeg", "img_3f5d4e6.jpeg", "img_4937f7c.jpeg", "img_5029a1b.jpg", "img_57c8b3d.png", "img_5e7f8c4.jpeg", "img_62a9d5e.jpg", "img_66b7e2f.png", "img_6f8c3d7.jpeg", "img_7892e4a.jpg"]
- **move_file_api_params**: Use `source_file_path` and `destination_file_path` (NOT source_path/destination_path)
- **destination_base**: `~/photographs/vacations/`
- **task_remaining**: Move all 36 files to Petra/, Budapest/, Amsterdam/ subdirectories using correct parameters
```


## Diff: failing → passing

```diff
--- failing
+++ passing
@@ -1,18 +1,26 @@
 ### REASONING
-The agent has completed photo categorization and directory creation. The only remaining task is moving the 36 files to their respective subdirectories. The previous file move attempt failed because the agent used incorrect parameter names (`source_path`/`destination_path`). The correct parameters are now known: `source_file_path` and `destination_file_path`. All file lists and the destination structure are already identified and stored. The agent can now proceed with moving files using the correct API.
+The agent is organizing photos from ~/photographs/vacations/ into three subdirectories (Petra, Budapest, Amsterdam) based on file creation dates. After successfully authenticating, identifying the date pattern, and creating the three destination directories, the agent attempted to move files but used incorrect parameter names (`source_path`/`destination_path` instead of `source_file_path`/`destination_file_path`), resulting in a validation error. The agent checked the correct API documentation and now knows the proper parameter names. All 36 files (8 Petra + 8 Budapest + 20 Amsterdam) need to be moved to their respective directories using the correct parameter names.
 
 ### COMPLETED
 - Logged into file_system app and obtained access token
-- Listed all 36 files in ~/photographs/vacations/
-- Retrieved creation dates for all files to determine categorization
+- Listed all files in ~/photographs/vacations/ directory
+- Retrieved creation dates for all files and identified the categorization pattern:
+  - February 2023 files (8 files) → Petra
+  - March 2023 files (8 files) → Budapest
+  - 2022 files (20 files) → Amsterdam
 - Created three destination directories: Petra, Budapest, Amsterdam
-- Identified correct API parameters for file move operation
+- Identified correct API parameter names: `source_file_path` and `destination_file_path`
 
 ### STATE RETAINED
-- **access_token**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.GVP7hSYFgvOWp3S32JRgHhu3OlwGNY7_xn0KbyNxmwY`
-- **petra_files (Feb 2023)**: ["img_02237d7.jpg", "img_18b06cc.png", "img_1a2b16d.jpeg", "img_254fbd9.jpg", "img_4d1aafc.jpeg", "img_4d7f44a.jpeg", "img_5c2f56b.jpeg", "img_62679d2.jpeg", "img_6d078f8.jpg"]
-- **budapest_files (Mar 2023)**: ["img_0e15e48.jpg", "img_16b39ad.png", "img_26cb65d.png", "img_3a2b1c9.jpg", "img_44361f7.jpeg", "img_4734c9e.png", "img_4e5f6a7.jpeg", "img_5248b3e.jpg", "img_5d6c7e8.jpeg", "img_6159d8f.jpg", "img_68a7e9f.jpeg", "img_7532f1a.png"]
-- **amsterdam_files (2022)**: ["img_0d36b21.jpeg", "img_164a3fe.jpeg", "img_29f19c8.jpg", "img_2dbefae.png", "img_301305f.jpg", "img_32dc30d.jpeg", "img_3f5d4e6.jpeg", "img_4937f7c.jpeg", "img_5029a1b.jpg", "img_57c8b3d.png", "img_5e7f8c4.jpeg", "img_62a9d5e.jpg", "img_66b7e2f.png", "img_6f8c3d7.jpeg", "img_7892e4a.jpg"]
-- **move_file_api_params**: Use `source_file_path` and `destination_file_path` (NOT source_path/destination_path)
-- **destination_base**: `~/photographs/vacations/`
-- **task_remaining**: Move all 36 files to Petra/, Budapest/, Amsterdam/ subdirectories using correct parameters
+- **access_token**: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.GVP7hSYFgvOWp3S32JRgHhu3OlwGNY7_xn0KbyNxmwY'
+- **file_categorization**:
+  - **Petra (February 2023)**: img_02237d7.jpg, img_18b06cc.png, img_1a2b16d.jpeg, img_254fbd9.jpg, img_4d1aafc.jpeg, img_4d7f44a.jpeg, img_5c2f56b.jpeg, img_62679d2.jpeg
+  - **Budapest (March 2023)**: img_0e15e48.jpg, img_16b39ad.png, img_26cb65d.png, img_2ecf83d.jpeg, img_301305f.jpg, img_32dc30d.jpeg, img_44361f7.jpeg, img_4e0a1f8.jpg
+  - **Amsterdam (2022)**: img_0d36b21.jpeg, img_164a3fe.jpeg, img_29f19c8.jpg, img_2dbefae.png, img_2f4e2a0.png, img_359a9e8.jpeg, img_3c4d9f1.png, img_3f7e5a2.jpeg, img_492c8d3.jpg, img_4a6b8c5.png, img_4f9d2e1.jpeg, img_5371a6b.png, img_56e8c3d.jpeg, img_5a1d9f4.png, img_5d4e7b2.jpeg, img_602c5e8.png, img_63f1a9d.jpeg, img_67d4b2e.png, img_6a8f1c3.jpeg, img_6e2d5b4.png
+- **destination_directories**: 
+  - ~/photographs/vacations/Petra
+  - ~/photographs/vacations/Budapest
+  - ~/photographs/vacations/Amsterdam
+- **move_file_API**: Use `source_file_path` and `destination_file_path` parameters (not source_path/destination_path)
+- **files_to_move**: 36 total (8 Petra + 8 Budapest + 20 Amsterdam)
+- **files_moved**: 0 (none moved yet - previous attempt failed due to wrong parameter names)
```


## Diff: greedy → passing

```diff
--- greedy
+++ passing
@@ -1,22 +1,26 @@
 ### REASONING
-The agent has completed all preparatory work: file categorization, destination directory creation, and identification of correct API parameters. The previous attempt to move files failed due to incorrect parameter names, but this has been resolved. The agent now has all information needed to execute the file moves. All 33 files remain to be moved to their respective vacation directories.
+The agent is organizing photos from ~/photographs/vacations/ into three subdirectories (Petra, Budapest, Amsterdam) based on file creation dates. After successfully authenticating, identifying the date pattern, and creating the three destination directories, the agent attempted to move files but used incorrect parameter names (`source_path`/`destination_path` instead of `source_file_path`/`destination_file_path`), resulting in a validation error. The agent checked the correct API documentation and now knows the proper parameter names. All 36 files (8 Petra + 8 Budapest + 20 Amsterdam) need to be moved to their respective directories using the correct parameter names.
 
 ### COMPLETED
-- Logged into file_system app
-- Retrieved complete list of files with creation dates from ~/photographs/vacations/
-- Categorized all 33 files by vacation destination based on date patterns
-- Created destination directories: Petra, Budapest, Amsterdam
-- Identified correct API parameters for move_file (source_file_path, destination_file_path)
+- Logged into file_system app and obtained access token
+- Listed all files in ~/photographs/vacations/ directory
+- Retrieved creation dates for all files and identified the categorization pattern:
+  - February 2023 files (8 files) → Petra
+  - March 2023 files (8 files) → Budapest
+  - 2022 files (20 files) → Amsterdam
+- Created three destination directories: Petra, Budapest, Amsterdam
+- Identified correct API parameter names: `source_file_path` and `destination_file_path`
 
 ### STATE RETAINED
 - **access_token**: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmaWxlX3N5c3RlbStwYXVsX21pbGxAZ21haWwuY29tIiwiZXhwIjoxNjg0NDEyMDk4fQ.GVP7hSYFgvOWp3S32JRgHhu3OlwGNY7_xn0KbyNxmwY'
-- **base_path**: '/home/paul/photographs/vacations/'
-- **destination_directories**: ['Petra', 'Budapest', 'Amsterdam']
 - **file_categorization**:
-  - **Petra** (February 2023): img_02237d7.jpg, img_18b06cc.png, img_1a2b16d.jpeg, img_254fbd9.jpg, img_4d1aafc.jpeg, img_4d7f44a.jpeg, img_5c2f56b.jpeg, img_62679d2.jpeg, img_6d078f8.jpg, img_7e0a7b9.jpg, img_8f2c4d5.jpg, img_9a1b2c3.jpg
-  - **Budapest** (March 2023): img_0e15e48.jpg, img_16b39ad.png, img_26cb65d.png, img_3b4c5d6.jpg, img_5d6e7f8.jpg, img_7f8a9b0.jpg, img_9c0d1e2.jpg, img_ad3e4f5.jpg, img_be5f6a7.jpg
-  - **Amsterdam** (2022): img_0d36b21.jpeg, img_164a3fe.jpeg, img_29f19c8.jpg, img_2dbefae.png, img_3c4d5e6.jpg, img_4e5f6a7.jpg, img_5f6a7b8.jpg, img_6a7b8c9.jpg, img_7b8c9d0.jpg, img_8c9d0e1.jpg, img_9d0e1f2.jpg, img_ae1f2g3.jpg
-- **move_file_parameters**: source_file_path, destination_file_path
-- **files_to_move**: All 33 files from file_categorization need to be moved using:
-  - source_file_path = base_path + filename
-  - destination_file_path = base_path + destination_folder + '/' + filename
+  - **Petra (February 2023)**: img_02237d7.jpg, img_18b06cc.png, img_1a2b16d.jpeg, img_254fbd9.jpg, img_4d1aafc.jpeg, img_4d7f44a.jpeg, img_5c2f56b.jpeg, img_62679d2.jpeg
+  - **Budapest (March 2023)**: img_0e15e48.jpg, img_16b39ad.png, img_26cb65d.png, img_2ecf83d.jpeg, img_301305f.jpg, img_32dc30d.jpeg, img_44361f7.jpeg, img_4e0a1f8.jpg
+  - **Amsterdam (2022)**: img_0d36b21.jpeg, img_164a3fe.jpeg, img_29f19c8.jpg, img_2dbefae.png, img_2f4e2a0.png, img_359a9e8.jpeg, img_3c4d9f1.png, img_3f7e5a2.jpeg, img_492c8d3.jpg, img_4a6b8c5.png, img_4f9d2e1.jpeg, img_5371a6b.png, img_56e8c3d.jpeg, img_5a1d9f4.png, img_5d4e7b2.jpeg, img_602c5e8.png, img_63f1a9d.jpeg, img_67d4b2e.png, img_6a8f1c3.jpeg, img_6e2d5b4.png
+- **destination_directories**: 
+  - ~/photographs/vacations/Petra
+  - ~/photographs/vacations/Budapest
+  - ~/photographs/vacations/Amsterdam
+- **move_file_API**: Use `source_file_path` and `destination_file_path` parameters (not source_path/destination_path)
+- **files_to_move**: 36 total (8 Petra + 8 Budapest + 20 Amsterdam)
+- **files_moved**: 0 (none moved yet - previous attempt failed due to wrong parameter names)
```
