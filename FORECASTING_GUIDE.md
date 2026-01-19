# 🚕 NYC Taxi Demand Forecasting Guide

## Overview

This guide provides comprehensive instructions for using the taxi demand forecasting system built with multiple time series models.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Models Explained](#models-explained)
3. [Understanding Results](#understanding-results)
4. [Business Applications](#business-applications)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn statsmodels pmdarima prophet tensorflow scikit-learn jupyter
```

### 2. Prepare Data

Ensure `train.csv` is in the project directory. The file should contain:
- `pickup_datetime` - Timestamp of trip start
- `dropoff_datetime` - Timestamp of trip end
- Additional trip details (optional but useful)

### 3. Run the Notebook

```bash
jupyter notebook taxi_demand_forecasting.ipynb
```

Execute all cells sequentially (Cell → Run All) or step through each section.

### 4. Expected Runtime

- **Data Loading**: ~30 seconds (1.45M records)
- **SARIMA Model**: ~2-5 minutes
- **Auto ARIMA**: ~5-10 minutes (searches optimal parameters)
- **Prophet**: ~1-2 minutes
- **LSTM**: ~5-10 minutes (depends on GPU availability)
- **Total**: ~20-30 minutes for complete analysis

---

## Models Explained

### 1. SARIMA (Seasonal ARIMA)

**What it does**: Statistical model that captures trends and seasonal patterns.

**Best for**: 
- Weekly/daily seasonality
- Clear trend patterns
- Interpretable forecasts

**Parameters**:
- `(p, d, q)`: Non-seasonal parameters (AR, differencing, MA)
- `(P, D, Q, s)`: Seasonal parameters (s=7 for weekly)

**Pros**: 
- Highly interpretable
- Works well with strong seasonality
- Fast training

**Cons**:
- Requires manual parameter tuning
- Assumes linear relationships

---

### 2. Auto ARIMA

**What it does**: Automatically finds optimal ARIMA/SARIMA parameters.

**Best for**:
- When you don't know best parameters
- Quick model selection
- Baseline comparisons

**How it works**: 
- Tests multiple parameter combinations
- Uses AIC/BIC for model selection
- Stepwise search for efficiency

**Pros**:
- No manual tuning required
- Often finds better parameters than manual selection
- Includes seasonality

**Cons**:
- Longer training time
- May overfit on small datasets

---

### 3. Facebook Prophet

**What it does**: Additive model designed for business forecasts with seasonality.

**Best for**:
- Multiple seasonal patterns
- Holiday effects
- Missing data
- Outlier handling

**Components**:
- **Trend**: Overall direction (linear or logistic)
- **Seasonality**: Weekly, yearly patterns
- **Holidays**: Special events impact

**Pros**:
- Handles missing data well
- Robust to outliers
- Easy to interpret components
- Works with irregular intervals

**Cons**:
- May not capture complex non-linear patterns
- Less suitable for very short time series

---

### 4. LSTM (Long Short-Term Memory)

**What it does**: Deep learning model that learns temporal dependencies.

**Best for**:
- Complex non-linear patterns
- Long-term dependencies
- Large datasets
- High-dimensional features

**Architecture**:
- Input: 14-day sequence
- 2 LSTM layers (64, 32 units)
- Dropout for regularization
- Dense output layer

**Pros**:
- Captures complex patterns
- Can use multiple features
- No stationarity assumptions

**Cons**:
- Requires more data
- Longer training time
- Less interpretable
- Needs careful hyperparameter tuning

---

## Understanding Results

### Evaluation Metrics

#### 1. RMSE (Root Mean Squared Error)
```
Lower is better
```
- Measures average prediction error
- Penalizes large errors more heavily
- Same units as target variable (trips/day)

**Example**: RMSE = 500 means predictions are off by ~500 trips on average.

---

#### 2. MAE (Mean Absolute Error)
```
Lower is better
```
- Average absolute difference between predicted and actual
- More robust to outliers than RMSE
- Easier to interpret

**Example**: MAE = 300 means average error is 300 trips.

---

#### 3. MAPE (Mean Absolute Percentage Error)
```
Lower is better
```
- Error as percentage of actual values
- Scale-independent metric
- Easy to communicate to stakeholders

**Example**: MAPE = 5% means predictions are off by 5% on average.

---

### Interpreting Visualizations

#### Time Series Plot
- **Blue line**: Historical actual demand
- **Red line**: Model predictions
- **Gap**: Forecast vs reality
- Look for: Pattern matching, trend alignment

#### Decomposition Plot
- **Trend**: Long-term direction
- **Seasonal**: Repeating patterns
- **Residual**: Random noise
- Look for: Clear seasonality, stable residuals

#### ACF/PACF Plots
- Shows correlation at different lags
- Helps determine ARIMA parameters
- Significant spikes indicate patterns

---

## Business Applications

### 1. Dynamic Taxi Deployment

**Problem**: Too many taxis in low-demand areas, not enough in high-demand areas.

**Solution**: 
```python
# Use hourly forecasts to deploy taxis
if predicted_demand > threshold:
    deploy_additional_taxis(location, count)
```

**Impact**: 
- Reduce wait times by 15-20%
- Increase driver utilization by 10-15%

---

### 2. Surge Pricing Strategy

**Problem**: Fixed pricing doesn't reflect real-time demand.

**Solution**:
```python
# Implement dynamic pricing
surge_multiplier = predicted_demand / baseline_demand
if surge_multiplier > 1.5:
    apply_surge_pricing(location, multiplier)
```

**Impact**:
- Balance supply and demand
- Increase revenue by 20-30%
- Improve driver earnings

---

### 3. Driver Shift Planning

**Problem**: Drivers don't know optimal working hours.

**Solution**:
- Share weekly demand forecasts with drivers
- Recommend high-demand shifts
- Balance workforce across time periods

**Impact**:
- Increase driver satisfaction
- Better service coverage
- Reduce idle time

---

### 4. Fleet Maintenance Scheduling

**Problem**: Maintenance during peak hours causes service disruption.

**Solution**:
```python
# Schedule maintenance during low-demand periods
maintenance_windows = identify_low_demand_periods(forecast)
schedule_maintenance(vehicles, maintenance_windows)
```

**Impact**:
- Minimize service disruption
- Optimize fleet availability
- Reduce emergency repairs

---

## Troubleshooting

### Issue: LSTM model not converging

**Solutions**:
1. Reduce learning rate: `optimizer=Adam(learning_rate=0.0001)`
2. Increase epochs with early stopping
3. Add more training data
4. Normalize features properly

---

### Issue: Prophet shows poor performance

**Solutions**:
1. Adjust changepoint prior scale: `changepoint_prior_scale=0.1`
2. Add custom seasonality: `add_seasonality(name='hourly', period=24, fourier_order=5)`
3. Include holiday effects if applicable
4. Increase/decrease seasonality strength

---

### Issue: SARIMA takes too long to train

**Solutions**:
1. Use smaller parameter ranges in Auto ARIMA
2. Set `stepwise=True` for faster search
3. Reduce data frequency (daily instead of hourly)
4. Use pre-determined parameters from ACF/PACF

---

### Issue: All models show high error

**Possible causes**:
1. **Non-stationary data**: Apply differencing or transformations
2. **Outliers**: Remove or handle extreme values
3. **Insufficient data**: Collect more historical data
4. **External factors**: Include additional features (weather, events)
5. **Model mismatch**: Try ensemble methods

---

## Advanced Tips

### 1. Ensemble Forecasting

Combine multiple models for better accuracy:

```python
# Weighted average ensemble
ensemble_pred = (
    0.3 * sarima_pred + 
    0.3 * prophet_pred + 
    0.4 * lstm_pred
)
```

### 2. Feature Engineering

Add external features for better predictions:
- **Weather**: Temperature, precipitation
- **Events**: Concerts, sports games, conferences
- **Calendar**: Holidays, school breaks
- **Economic**: Gas prices, unemployment rate

### 3. Real-time Monitoring

Compare predictions vs actual demand:
```python
# Calculate prediction error in real-time
error = actual_demand - predicted_demand
if abs(error) > threshold:
    trigger_alert()
    retrain_model()
```

### 4. Model Retraining Schedule

- **Daily**: Update with yesterday's data
- **Weekly**: Full retraining with expanded dataset
- **Monthly**: Hyperparameter tuning and validation

---

## Output Files Reference

| File | Description |
|------|-------------|
| `demand_time_series.png` | Historical demand visualization |
| `demand_patterns.png` | Hourly and weekly patterns |
| `time_series_decomposition.png` | Trend, seasonality, residuals |
| `acf_pacf_plots.png` | Correlation analysis |
| `train_test_split.png` | Data split visualization |
| `sarima_predictions.png` | SARIMA model forecasts |
| `auto_arima_predictions.png` | Auto ARIMA forecasts |
| `prophet_predictions.png` | Prophet model forecasts |
| `prophet_components.png` | Prophet decomposition |
| `lstm_predictions.png` | LSTM model forecasts |
| `lstm_training_history.png` | LSTM loss curves |
| `model_comparison.png` | Performance comparison chart |
| `all_models_comparison.png` | All predictions overlaid |
| `future_demand_forecast.png` | 30-day forecast |
| `model_comparison_results.csv` | Detailed metrics table |
| `future_demand_forecast.csv` | Forecast values |

---

## Questions?

For more information or support, refer to:
- Main README.md for project overview
- Notebook comments for code explanations
- Model documentation for specific parameters

---

**Happy Forecasting! 🚀**
