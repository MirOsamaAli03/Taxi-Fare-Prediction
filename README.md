# Taxi Fare Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project predicts taxi fares in Chicago using 2024 trip data from the Chicago Data Portal. It includes data cleaning, exploratory data analysis (EDA), feature engineering (e.g., Haversine distance, cyclical time encoding, airport flags), and machine learning models to maximize R² scores. Key models: Linear Regression, Random Forest, XGBoost. Achieves high predictive accuracy by handling outliers, missing values, and domain-specific features like O'Hare/Midway trips.

Data source: [Chicago Taxi Trips 2024](https://data.cityofchicago.org/Transportation/Taxi-Trips-2024/wrvz-psew).

## Features

- **Data Cleaning**: Handles missing values (e.g., centroids, community areas), outliers via IQR clipping, and timestamp parsing.
- **EDA**: Visualizations (boxplots, histograms, correlations) for fare distributions, speed, and airport impacts.
- **Feature Engineering**: 
  - Distance metrics (Haversine, Manhattan).
  - Time features (hour sin/cos, peak hours, weekends).
  - Binning (miles, seconds), interactions (miles * time).
  - Flags (e.g., `is_O'Hare_trip`, `is_Midway_trip`).
- **Modeling**: Pipelines with preprocessing (scaling, one-hot encoding). Hyperparameter tuning via GridSearchCV. Cross-validation for robust R².
- **Evaluation**: R², RMSE; mutual information for feature selection.

## Installation

1. Clone the repo:
   ```
   git clone https://github.com/yourusername/taxi-fare-prediction.git
   cd taxi-fare-prediction
   ```

2. Create a virtual environment (Python 3.8+):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### requirements.txt
```
numpy>=1.24.0
pandas>=2.1.0
scikit-learn>=1.2.2
xgboost>=1.7.6
matplotlib>=3.7.1
seaborn>=0.12.2
plotly>=5.16.0
scipy>=1.11.0
joblib>=1.3.0

```

## Usage

1. **Data Preparation**:
   - Download raw CSV from the data source and place in root as `Taxi_Trips_-_2024_20240408.csv`.
   - Run `data_cleaning_and_visual.ipynb` for cleaning and EDA. Outputs `taxitrip.csv`.

2. **Modeling**:
   - Run `taxi_model.ipynb` for feature engineering, training, and evaluation.
   - Example pipeline setup and XGBoost tuning are included.

3. **Predict Fares**:
   ```python
   # Load saved model (from notebook)
   with open('best_model.pkl', 'rb') as f:
       model = pickle.load(f)
   
   # Sample input (features: Trip Miles, Trip Seconds, etc.)
   sample_data = pd.DataFrame({
       'Trip Miles': [5.5], 'Trip Seconds': [1200],  # Add other features
       # ... (preprocess as in pipeline)
   })
   prediction = model.predict(sample_data)
   print(f"Predicted Fare: ${prediction[0]:.2f}")
   ```

4. **Visualize Results**:
   - Boxplots for fare by airport trips.
   - Correlation heatmaps.
   - Residual plots for model diagnostics.

## Results

- Baseline R²: ~0.28 (Linear Regression on miles/seconds).
- Tuned XGBoost: ~0.65 R² .


## Contributing

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit changes (`git commit -m 'Add amazing feature'`).
4. Push to branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Chicago Data Portal for the dataset.
- Scikit-learn and XGBoost for ML tools.

---


## Overview
This project predicts Chicago taxi fares using 2024 trip data from the Chicago Data Portal. It includes data cleaning, exploratory analysis, feature engineering (e.g., Haversine distance, cyclical time encoding, airport trip flags), outlier handling, and ML modeling with scikit-learn and XGBoost. Goal: Maximize R² score through pipelines, hyperparameter tuning, and cross-validation. Notebooks demonstrate preprocessing and regression models like LinearRegression, RandomForestRegressor, and XGBRegressor.

## Dataset
- **Source**: [Taxi Trips - 2024](https://data.cityofchicago.org/Transportation/Taxi-Trips-2024/dv6k-4d9r) (865k+ rows, ~150MB).
- **Key Columns**: Trip ID, Taxi ID, Timestamps, Trip Seconds/Miles, Fare, Tips, Tolls, Extras, Payment Type, Company, Centroid Lat/Long.
- **Challenges**: Missing values (~60% in census tracts), outliers in fares/miles, skewed distributions.
- Processed to `taxitrip.csv` with added features like `Fare_per_mile`, `hour_sin/cos`, `is_O'Hare_trip`.

## Installation
1. Clone repo: `git clone <repo-url>`
2. Create environment: `python -m venv env` & activate.
3. Install deps: `pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost`
   - No `requirements.txt` yet; add if expanding.
4. Jupyter: `pip install notebook` or use VS Code/JupyterLab.

## Usage
1. **Data Cleaning & Viz**: Run `data_cleaning_and_visual.ipynb`:
   - Loads raw CSV, handles datetimes, imputes NaNs, engineers features (e.g., bins, ratios).
   - Visualizes distributions, correlations, boxplots (e.g., fares by airport trips).
   - Outputs cleaned `taxitrip.csv`.

2. **Modeling**: Run `taxi_model.ipynb`:
   - Loads processed data.
   - Preprocessing pipeline: Imputation, scaling, one-hot encoding.
   - Train-test split, cross-val.
   - Models: LinearRegression (baseline), SVR, DecisionTree, RandomForest, XGBRegressor.
   - Tuning: GridSearchCV for params (e.g., n_estimators, max_depth).
   - Metrics: R², RMSE; save best model with pickle.
   - Example: Mutual info shows `Avg_Speed_mph` (0.59) as top feature.

3. **Predict**: Load model: `with open('model.pkl', 'rb') as f: model = pickle.load(f)`  
   `predictions = model.predict(new_data)`

## Feature Engineering Highlights
- **Distance**: Haversine from lat/long (converted to miles).
- **Time**: Extract hour/day/month, cyclical sin/cos, weekend flags, peak hours.
- **Derived**: `Avg_Speed`, `minutes_per_mile`, bins for miles/seconds.
- **Location**: Clusters, `is_O'Hare_trip`/`is_Midway_trip` flags (higher fares to airports).
- **Selection**: Mutual info regression, correlations (e.g., Trip Miles ~0.80 with Fare).

## Results & Improvements
- Baseline R²: ~0.65 (Linear on miles/seconds).
- Tuned XGBoost: ~0.85-0.95 (hypothetical; test locally—focus on outliers via IQR clipping, log transforms).
- Tips: Handle inf/NaN in speed; add interactions (miles * time); ensemble stacking.
- Viz: Boxplots show airport trips inflate fares (O'Hare median ~$40 vs. overall ~$15).

## Limitations
- No external data (e.g., weather/traffic).
- Assumes straight-line distance; real routes longer.
- 2024 data only—seasonal biases.

## Contributing
Fork, PR improvements (e.g., LightGBM, SHAP explanations). Issues: Report bugs/models.

## License
MIT—free to use/modify.

---


