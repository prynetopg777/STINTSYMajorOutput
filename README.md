# STINTSYMajorOutput

Personalized anime rating prediction using multiple supervised learning models.

## Notebooks (Run in Order)

1. **01_AnimePref.ipynb** — Load & explore raw MyAnimeList data
2. **02_FilterData.ipynb** — Clean, filter, and engineer features
3. **03_LinearRegression.ipynb** — Train linear regression baseline
4. **04_RandomForest.ipynb** — Train random forest model
5. **05_NeuralNetwork.ipynb** — Train hybrid neural network with embeddings

## Project Structure

```
processed-data/
├── anime.csv                           # Preprocessed anime metadata
├── model_evaluation.csv                # Model metrics (long format)
├── model_evaluation_values_wide.csv    # Model metrics (wide format)
├── splits/
│   ├── anime_train.csv
│   ├── anime_val.csv
│   └── anime_test.csv
|raw-dataset/
├── AnimeList.csv
├── UserAnimeList.csv
└── UserList.csv

```

## Models Trained

- **Linear Regression** — baseline (Ridge regularization)
- **Random Forest** — ensemble method
- **Neural Network** — hybrid wide & deep architecture with embeddings

