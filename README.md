# DATA-VISUALIZATION-URBAN-MOBILITY-ANALYSIS
This comprehensive analysis of 1.45 million NYC yellow taxi trips delivers actionable insights for operational optimization and strategic planning.

## 📊 Project Components

### 1. **NYC TAXI VISU.ipynb**
Comprehensive exploratory data analysis and visualization of NYC taxi trips including:
- Trip patterns and distributions
- Temporal analysis (hourly, daily, weekly, monthly)
- Geographic analysis and heatmaps
- Speed and fare analysis
- Feature engineering

### 2. **taxi_demand_forecasting.ipynb** 🆕
Advanced time series forecasting to predict taxi demand using multiple models:

#### Models Implemented:
1. **SARIMA (Seasonal ARIMA)** - Traditional statistical approach with seasonal components
2. **Auto ARIMA** - Automated parameter selection for optimal ARIMA configuration
3. **Facebook Prophet** - Additive model designed for forecasting with seasonality
4. **LSTM Neural Network** - Deep learning approach for sequence prediction

#### Key Features:
- Hourly and daily demand aggregation
- Comprehensive time series decomposition
- Stationarity testing (ADF test)
- ACF/PACF analysis
- Model comparison and performance evaluation
- 30-day future demand predictions
- Business insights and recommendations

#### Evaluation Metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn statsmodels pmdarima prophet tensorflow scikit-learn
```

### Usage

1. **Data Exploration and Visualization:**
   ```bash
   jupyter notebook "NYC TAXI VISU.ipynb"
   ```

2. **Demand Forecasting:**
   ```bash
   jupyter notebook taxi_demand_forecasting.ipynb
   ```

## 📈 Results

The time series models provide:
- Accurate demand predictions for resource planning
- Identification of peak demand periods
- Weekly and daily seasonality patterns
- Future demand forecasts for operational optimization

## 🎯 Business Applications

- **Resource Allocation**: Deploy taxis based on predicted demand
- **Dynamic Pricing**: Implement surge pricing during high-demand periods
- **Driver Scheduling**: Optimize shift patterns using demand forecasts
- **Fleet Management**: Plan maintenance during low-demand periods
- **Demand Response**: Monitor and adjust to real-time demand variations

## 📁 Dataset

- **Source**: NYC Yellow Taxi Trip Data (train.csv)
- **Records**: 1,458,644 trips
- **Time Period**: ~6 months
- **Features**: Pickup/dropoff datetime, locations, passenger count, trip distance, fares

## 🛠️ Technical Stack

- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Statistical Analysis**: Statsmodels
- **Time Series**: ARIMA, SARIMA, Auto ARIMA, Prophet
- **Deep Learning**: TensorFlow/Keras (LSTM)
- **Machine Learning**: Scikit-learn

## 📊 Output Files

- `demand_time_series.png` - Time series visualization
- `demand_patterns.png` - Hourly and daily patterns
- `time_series_decomposition.png` - Trend, seasonality, residuals
- `model_comparison.png` - Performance comparison
- `future_demand_forecast.png` - 30-day forecast
- `model_comparison_results.csv` - Detailed metrics
- `future_demand_forecast.csv` - Forecast data 
