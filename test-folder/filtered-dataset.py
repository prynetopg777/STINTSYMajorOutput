"""
filtered-dataset.py
===================
Merges AnimeList + UserList + UserAnimeList into a single ML-ready dataframe.

Pipeline:
    1. Load the 3 cleaned CSVs (chunked for the large UserAnimeList)
    2. Filter UserAnimeList to only completed/scored entries (my_status == 2, my_score > 0)
    3. Merge: UserAnimeList ← UserList (on username) ← AnimeList (on anime_id)
    4. Engineer new features (completion_ratio, score_vs_user_mean)
    5. Select & export the final feature set (9 ML features)

Output: filtered-dataset/master_df.csv  (and a .pkl for faster loading)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "main-dataset"
OUT_DIR     = "filtered-dataset"
os.makedirs(OUT_DIR, exist_ok=True)

ANIME_PATH  = os.path.join(DATA_DIR, "anime_cleaned.csv")
USERS_PATH  = os.path.join(DATA_DIR, "users_cleaned.csv")
ALIST_PATH  = os.path.join(DATA_DIR, "animelists_cleaned.csv")

# ── Step 1 - Load anime metadata ───────────────────────────────────────────────
print("[1/5] Loading anime metadata …")
anime_df = pd.read_csv(ANIME_PATH, low_memory=False)
print(f"      anime_cleaned  : {anime_df.shape[0]:,} rows × {anime_df.shape[1]} cols")

# Keep only the anime features used in ML (+ identifiers)
ANIME_KEEP = [
    "anime_id",
    "title",    # kept for display/reference only, not a model feature
    "type",     # TV, Movie, OVA, Special, ONA, Music
    "episodes", # series length
    "genre",    # multi-label → multi-hot encoded later
    "source",   # Manga, Light Novel, Original, …
]
anime_df = anime_df[[c for c in ANIME_KEEP if c in anime_df.columns]]

# ── Step 2 - Load user profiles ────────────────────────────────────────────────
print("[2/5] Loading user profiles …")
users_df = pd.read_csv(USERS_PATH, low_memory=False)
print(f"      users_cleaned  : {users_df.shape[0]:,} rows × {users_df.shape[1]} cols")

# User age at time of data collection (approximate)
REFERENCE_YEAR = 2018   # dataset was collected ~2018
users_df["birth_date"] = pd.to_datetime(users_df["birth_date"], errors="coerce")
users_df["user_age"] = REFERENCE_YEAR - users_df["birth_date"].dt.year

# Keep only the user features used in ML (+ identifier)
USERS_KEEP = [
    "username",
    "gender",           # demographic taste signal
    "user_age",         # demographic taste signal
    "stats_mean_score", # user's personal average rating (taste calibration)
]
users_df = users_df[[c for c in USERS_KEEP if c in users_df.columns]]

# ── Step 3 - Load & filter UserAnimeList (large file, read in chunks) ──────────
print("[3/5] Loading & filtering user-anime interactions …")
print("      (This may take a minute - the file is ~2 GB)")

CHUNK_SIZE  = 500_000
VALID_STATUS = 2       # 2 = Completed
MIN_SCORE    = 1       # exclude "no score" (0)

chunks = []
chunk_count = 0
for chunk in pd.read_csv(ALIST_PATH, chunksize=CHUNK_SIZE, low_memory=False):
    chunk_count += 1
    filtered = chunk[
        (chunk["my_status"] == VALID_STATUS) &
        (chunk["my_score"]  >= MIN_SCORE)
    ]
    chunks.append(filtered)
    if chunk_count % 5 == 0:
        print(f"      … processed {chunk_count * CHUNK_SIZE:,} rows so far")

interactions_df = pd.concat(chunks, ignore_index=True)
print(f"      interactions (completed + scored): {interactions_df.shape[0]:,} rows")

ALIST_KEEP = [
    "username",
    "anime_id",
    "my_score",            # used to derive target (is_recommended) and score_vs_user_mean
    "my_watched_episodes", # used to derive completion_ratio
]
interactions_df = interactions_df[[c for c in ALIST_KEEP if c in interactions_df.columns]]

# ── Step 4 - Three-way merge ───────────────────────────────────────────────────
print("[4/5] Merging datasets …")

# interactions ←→ users  (on username)
df = interactions_df.merge(users_df, on="username", how="inner")
print(f"      after join with users : {df.shape[0]:,} rows")

# df ←→ anime  (on anime_id)
df = df.merge(anime_df, on="anime_id", how="inner")
print(f"      after join with anime : {df.shape[0]:,} rows")

# ── Step 5 - Derive the two behavioral features ────────────────────────────────
# completion_ratio: fraction of episodes the user actually watched [0, 1]
df["completion_ratio"] = np.where(
    df["episodes"] > 0,
    df["my_watched_episodes"] / df["episodes"],
    np.nan
)
df["completion_ratio"] = df["completion_ratio"].clip(0, 1)

# score_vs_user_mean: how much this score deviates from the user's own average
df["score_vs_user_mean"] = df["my_score"] - df["stats_mean_score"]

# Drop the raw helper column — no longer needed as a standalone feature
df.drop(columns=["my_watched_episodes"], inplace=True)

# ── Step 6 - Column summary ────────────────────────────────────────────────────
print("\n[5/5] Final feature summary:")
print(f"      Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
print(df.dtypes.to_string())

print("\nSample rows:")
print(df.head(3).to_string())

print("\nMissing value counts:")
print(df.isnull().sum().to_string())

# ── Save ───────────────────────────────────────────────────────────────────────
OUT_CSV = os.path.join(OUT_DIR, "master_df.csv")
OUT_PKL = os.path.join(OUT_DIR, "master_df.pkl")

print(f"\nSaving to {OUT_CSV} …")
df.to_csv(OUT_CSV, index=False)

print(f"Saving to {OUT_PKL} …")
df.to_pickle(OUT_PKL)

print("\nDone! ✓")
print(f"  CSV  → {OUT_CSV}")
print(f"  PKL  → {OUT_PKL}")
print(f"\nFEATURE SUMMARY (9 ML features + identifiers + target)")
print("=" * 60)
print("""
IDENTIFIERS (excluded from model input)
  username, anime_id, title

TARGET VARIABLE (derive before training)
  my_score → is_recommended  (1 if my_score >= 8, else 0)

ML FEATURES (9)
  User demographic (2):
    gender            - Male / Female / NaN → one-hot encode
    user_age          - approximate age (int); impute median for NaN

  Anime metadata (3 + genre):
    type              - TV / Movie / OVA / Special → one-hot encode
    episodes          - series length (int); impute median for NaN
    source            - Manga / Light Novel / Original → one-hot encode
    genre             - multi-label string → MultiLabelBinarizer

  Behavioral / derived (3):
    stats_mean_score  - user's personal average rating (float)
    completion_ratio  - my_watched_episodes / episodes [0.0 – 1.0]
    score_vs_user_mean - my_score minus stats_mean_score (float)

NOTE: genre expands into multiple binary columns after encoding.
      Treat it as 1 conceptual feature occupying N binary columns.
  user_age          - approximate age
  country           - simplified from location field
  years_on_platform - how long the user has been on MAL
  user_completed    - total anime completed (experience proxy)
  user_dropped      - tendency to drop anime
  user_days_spent_watching  - engagement depth
  stats_mean_score  - user's personal average rating (taste calibration)
  stats_rewatched   - rewatch habit

INTERACTION FEATURES (derived)
  completion_ratio  - my_watched_episodes / episodes  [0-1]
  score_vs_user_mean - my_score minus user's own average (relative taste)
  my_rewatching     - was this a rewatch?

FEATURES TO EXCLUDE / HANDLE CAREFULLY
  username, user_id, anime_id  - identifiers, not features
  title, title_*               - free text, needs NLP or drop
  my_watched_episodes          - can be used only with care (leakage boundary)
  score (community avg)        - leaks target if reused naively; use with caution
""")
