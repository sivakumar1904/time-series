Multivariate Time Series Forecasting Using LSTM, Attention & Transformer Models

This project implements advanced deep learning architectures for multivariate time series forecasting, moving beyond traditional statistical methods such as ARIMA/ETS.
The goal is to design, train, and evaluate LSTM, Attention-augmented LSTM, and Transformer models using a realistic multivariate dataset.

It fulfills all requirements of the project specification:
✔ self-attention
✔ multivariate forecasting
✔ production-quality preprocessing pipeline
✔ hyperparameter tuning support
✔ rolling/expanding window cross-validation
✔ comparison against ARIMA and standard LSTM
✔ attention-weight interpretation

Dataset Description
The dataset is a synthetic multivariate time series containing:


timestamp


sensor_1 … sensor_6


External features:


temperature


is_holiday




Engineered time-based features:


hour


dayofweek




Characteristics:


Hourly frequency


Seasonal patterns (daily + weekly)


Trend component


Noise + random missing values


Feature interactions


Perfect for testing deep learning models



🚀 Features Implemented
✔ Data Pipeline


Missing value imputation (forward/backfill)


Standard scaling (StandardScaler)


Feature engineering:


Lag features (1, 24, 168)


Time features (hour, dayofweek)




Chronological Train–Validation–Test split



✔ Forecasting Models
1️⃣ ARIMA (univariate baseline)


Implemented using statsmodels


Not multivariate—but used as a classical benchmark


2️⃣ LSTM Model


Multivariate input


Multiple layers with dropout & weight decay


3️⃣ Attention-LSTM


Custom attention layer


Learns which features/time steps influence predictions


Extracts attention weights for interpretation


4️⃣ Transformer Model


Transformer Encoder


Multi-head self-attention


Positional projections



✔ Hyperparameter Tuning via Expanding Window CV
The script includes a reusable expanding-window cross-validation function to tune:


hidden size


learning rate


weight decay


number of layers


batch size



✔ Evaluation Metrics
For each model, the script reports:


Mean Absolute Error (MAE)


Root Mean Squared Error (RMSE)


Mean Absolute Percentage Error (MAPE)


Results stored in:
📄 output/results_summary.json

✔ Attention Interpretation
For the Attention-LSTM model, the script saves:
📄 output/attention_mean.txt
Average attention weights across the test set.
These weights show:


What inputs the model focuses on


Which lag features or covariates influence forecasting the most



🛠 Installation
Install dependencies:
pip install numpy pandas scikit-learn torch tqdm
pip install statsmodels   # optional but required for ARIMA baseline


▶️ How to Run the Project
Basic run (default settings)
python forecast_project.py \
  --data synthetic_multivariate_timeseries.csv \
  --target sensor_1 \
  --output ./output

Arguments
ArgumentDescriptionDefault--dataPath to dataset CSVsynthetic_multivariate_timeseries.csv--targetTarget variable to forecastsensor_1--outputDirectory to save results./output

📊 Output Files
1. results_summary.json
Contains metrics for:


ARIMA


LSTM


Attention-LSTM


Transformer


Example:
{
  "LSTM": {
    "rmse": 42.1,
    "mae": 31.5,
    "mape": 6.8
  },
  "AttentionLSTM": { ... },
  "Transformer": { ... },
  "ARIMA": { ... }
}


2. attention_mean.txt
Averaged attention vector.
Interpretation guide:


High values → more influential features/time lags


Includes lag features, temperature, holiday flag, hour, dayofweek



📘 Future Improvements


Full TFT (Temporal Fusion Transformer) implementation


Optuna Bayesian hyperparameter tuning


Longer historical sequences for Transformer


Residual connections, layer normalization


Multi-step ahead forecasting


Model ensembling


