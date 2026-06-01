# Form-prior drop case — `27e1026_2` / `general_task_aware` / round `CK`

## Token under analysis

- **category**: `phone`
- **value (truncated)**: `2023-04-23`
- **original-trajectory context**:
  > ..."    },    {     "id": 6,     "name": "Seraphina Dawn"    }   ],   "added_at": "**2023-04-23**T10:52:29"  },  {   "song_id": 57,   "title": "Silver Lining",   "album_id": 11,...

## Headline numbers

| metric | value |
|---|---|
| sample preservation count | **4 / 8** |
| pass rate WITH token preserved  | **50%** |
| pass rate WITHOUT token (dropped) | **0%** |
| **pass-rate gap (with − without)** | **+50 pp** |
| greedy preserved this token? | False |
| greedy pass | True |
| full-context baseline | False |
| sample preserve pattern (submission order) | `[0, 0, 1, 1, 1, 0, 0, 1]` |
| sample pass pattern | `[0, 0, 1, 0, 1, 0, 0, 0]` |

## Task instruction

> What is the title of the newest released song in my Spotify account from across my song, album and playlist libraries?


## v5 / v7 narrative for this case

The token `2023-04-23` is present in the original AppWorld trajectory; under K=2 recompression in the `general_task_aware` family, **4 of 8 stochastic samples drop it**, while 4 preserve it. Samples that preserve the token pass downstream at 50%; samples that drop it pass at 0% (+50 pp gap). This is the v5 'recovered-then-dropped' phenomenon and the v7 'form-prior conditions on fact_type, not need_label' phenomenon manifesting at sample-distribution level: the same compressor applied to the same input drops critical functional content with non-trivial probability, and best-of-N selection exploits this variance.


## Greedy compressed — 804 chars (candidate `27e1026_2__general_task_aware__greedy`)

- token `phone` preserved: **False**

```
- Spotify credentials: username 'jes.mill@gmail.com', password '7w0A*Vw'
- Access token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzcG90aWZ5K2plcy5taWxsQGdtYWlsLmNvbSIsImV4cCI6MTY4NDQxMjA5OH0.hn6fnManbFM0p2iQt5LOVJR5jSF7IoPOY2_Zb72-nS0'
- Retrieved three libraries (page_limit=20):
  - Song library: song_id, title, album_id, duration, artists, added_at
  - Album library: album_id, title, genre, artists, rating, release_date, song_ids, added_at
  - Playlist library: playlist_id, title, is_public, rating, created_at, song_ids
- Output truncated; need to fetch all pages or analyze full data
- Task: Find newest released song by comparing release dates across all three libraries (song added_at, album release_date, playlist created_at combined with song_ids to get individual song release info)
```

## Passing sample (token preserved) — 949 chars (candidate `27e1026_2__general_task_aware__sample_03`)

- token `phone` preserved: **True**
- snippet around token: > ...e_limit=20: - Song library: includes "Reflections in Pool of Regret" (added_at: **2023-04-23**), "Silver Lining" (added_at: 2023-04-26), "The Haunting Echo of Goodbye" (added...

```
Task: Find title of newest released song in Spotify account from song library, album library, and playlist library.

Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzcG90aWZ5K2plcy5taWxsQGdtYWlsLmNvbSIsImV4cCI6MTY4NDQxMjA5OH0.hn6fnManbFM0p2iQt5LOVJR5jSF7IoPOY2_Zb72-nS0

Partial data retrieved with page_limit=20:
- Song library: includes "Reflections in Pool of Regret" (added_at: 2023-04-23), "Silver Lining" (added_at: 2023-04-26), "The Haunting Echo of Goodbye" (added_at: 2023-02-07)
- Album library: includes "Echo Chamber Chronicles" (release_date: 2022-10-29), "Neon Echoes" (release_date: 2021-03-23)
- Playlist library: includes "Retro Rewind: 80's & 90's Mix" (created_at: 2023-05-04), "Vocal Virtuosity: A Cappella Excellence" (created_at: 2023-03-24)

Note: Data truncated. Need to retrieve full libraries to find the newest released song. Compare:
- release_date for albums
- added_at for songs
- created_at for playlists
```

## Failing sample (token dropped, representative) — 782 chars (candidate `27e1026_2__general_task_aware__sample_01`)

- token `phone` preserved: **False**

```
Task: Find the newest released song in Spotify account from song library, album library, and playlist library.

Authentication completed. Access token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJzcG90aWZ5K2plcy5taWxsQGdtYWlsLmNvbSIsImV4cCI6MTY4NDQxMjA5OH0.hn6fnManbFM0p2iQt5LOVJR5jSF7IoPOY2_Zb72-nS0

Data retrieved:
- Song library: most recent added_at: 2023-04-26 (song_id 57 "Silver Lining")
- Album library: most recent added_at: 2023-05-15 (album_id 1)
- Playlist library: most recent: 2023-05-04 (playlist_id 133)

Note: To get individual song release dates from albums, use album details API. To get song release dates from playlists, use playlist details API.

Clarification pending: Does "newest released" mean release date (original release) or date added to library?
```