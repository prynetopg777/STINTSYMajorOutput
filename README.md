# STINTSYMajorOutput

A personalized anime recommendation system using supervised learning to predict user ratings on the MyAnimeList dataset.

## Prerequisites

Download the required datasets from Kaggle:
- **Source**: https://www.kaggle.com/datasets/azathoth42/myanimelist

**Required files**:
- `AnimeList.csv` (~14k-17k anime with metadata)
- `UserList.csv` (~300k users with profile stats)
- `UserAnimeList.csv` (~80M user-anime interactions)

**Setup**: Update `DATA_DIR` variable in notebooks to point to your downloaded data location.

---

## Pipeline Overview

This project follows a 3-stage machine learning pipeline:

```
Raw Data → Data Processing → Model Training & Evaluation
```

### Stage 1: Data Loading & Exploratory Analysis
### Stage 2: Data Filtering & Preprocessing  
### Stage 3: Linear Regression Model Training

---

## Step-by-Step Execution Guide

### **Step 1: Run `01_AnimePref.ipynb` — Anime Preference Data Loading & EDA**

**Purpose**: Load raw Kaggle data and perform exploratory data analysis

**What it does**:
- Loads `AnimeList.csv`, `UserList.csv`, and `UserAnimeList.csv` from your local data directory
- Validates data schemas and checks for missing values
- Computes dataset statistics (# users, # anime, # ratings, sparsity)
- Creates visualizations of user behavior, anime metadata distributions, and rating patterns
- Explores feature correlations and identifies key trends in anime preferences

**Output**: 
- Cleaned and validated base datasets in memory
- EDA plots and statistical summaries (displayed in notebook)

**Actions**:
1. Update `DATA_DIR` to point to your Kaggle data folder
2. Run all cells in order
3. Review the EDA outputs to understand data characteristics

---

### **Step 2: Run `02_FilterData.ipynb` — Data Preprocessing & Feature Engineering**

**Purpose**: Filter, clean, and engineer features from raw data to create a modeling-ready dataset

**What it does**:
- Loads validated datasets from Step 1
- Filters interactions by quality criteria (e.g., removes low-signal interactions)
- Performs data cleaning and deduplication
- Engineers user and anime interaction features
- Creates preprocessed interaction table with features ready for modeling
- Saves processed data as `animepref_final_interactions.csv` (and optionally `.parquet`)

**Output**:
- `processed-data/animepref_final_interactions.csv` — Main modeling dataset
- `processed-data/splits/` — Train/validation/test splits (created)

**Actions**:
1. Ensure Step 1 has completed successfully
2. Run all cells in order
3. Verify that `processed-data/animepref_final_interactions.csv` is created (~100MB-500MB typical size)

---

### **Step 3: Run `03_LinearRegression.ipynb` — Model Training & Evaluation**

**Purpose**: Train a linear regression model to predict anime ratings and evaluate performance

**What it does**:
- Loads preprocessed interactions from Step 2 (`animepref_final_interactions.csv`)
- Splits data chronologically into train (70%), validation (15%), and test (15%) sets
- Trains a Linear Regression model with Ridge regularization (α tuning via GridSearchCV)
- Evaluates model on validation and test sets using:
  - **MAE** (Mean Absolute Error) — average point error
  - **RMSE** (Root Mean Squared Error) — penalizes large errors
  - **R²** (Coefficient of Determination) — variance explained
- Visualizes predictions, residuals, and feature importance

**Output**:
- Trained model metrics and evaluation plots
- Feature coefficients showing which factors most influence predicted ratings

**Actions**:
1. Ensure Steps 1 & 2 have completed successfully
2. Run all cells in order
3. Review the performance metrics and error analysis

---

## Full Pipeline Execution

Run the notebooks in this exact order:

1. `01_AnimePref.ipynb` — Load & explore raw data
2. `02_FilterData.ipynb` — Process & engineer features  
3. `03_LinearRegression.ipynb` — Train & evaluate model

**Estimated runtime**: 
- Step 1: ~10-30 minutes (depends on data size and EDA depth)
- Step 2: ~5-15 minutes (data filtering & feature engineering)
- Step 3: ~5-10 minutes (model training & evaluation)

---

## Project Structure

```
STINTSYMajorOutput/
├── 01_AnimePref.ipynb                          # Stage 1: EDA
├── 02_FilterData.ipynb                         # Stage 2: Preprocessing
├── 03_LinearRegression.ipynb                   # Stage 3: Modeling
├── processed-data/
│   ├── animepref_final_interactions.csv        # Output from Step 2
│   └── splits/                                 # Train/val/test data
└── README.md                                   # This file
```

---

## Notes

- **Data privacy**: Ensure you have proper access/licenses for the Kaggle dataset
- **Memory requirements**: The full MyAnimeList dataset can be large; adjust data filters in Step 2 if needed
- **Reproducibility**: All notebooks use `SEED = 42` for consistent random state 
