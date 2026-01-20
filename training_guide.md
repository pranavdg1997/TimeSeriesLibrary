# Time Series Training Guide

Welcome to the **Universal Time Series** training guide. This document explains fundamental time series concepts and provides a framework for selecting the right model for your data.

---

## 1. Types of Time Series Data

Understanding your data structure is the first step in choosing a backend.

| Type | Description | Example |
| :--- | :--- | :--- |
| **Univariate** | A single target variable measured over time. | Daily temperature in a single city. |
| **Multivariate** | Multiple related variables measured over time. | Temperature, humidity, and wind speed in one city. |
| **Single Series** | One unique entity measured over time. | Stock price of 'AAPL'. |
| **Panel Data** | Multiple entities of the same type measured over time. | Sales of 1,000 different products across 50 stores. |

---

## 2. Commonly Used Terms

| Term | Also Known As | Definition |
| :--- | :--- | :--- |
| `ds` | Timestamp / Index | The time column. Must be datetime or convertible strings. |
| `y` | Target / Ground Truth | The value you are trying to forecast. |
| **Horizon** | `prediction_length` | How many steps into the future you want to predict. |
| **Frequency** | `freq` | The interval between data points (e.g., 'D' for Daily, 'H' for Hourly). |
| **Covariates** | Exogenous Variables | External features that influence the target (e.g., price, weather). |
| **Regressors** | Predictors | External features that are *known* for the future periods. |
| **Seasonality** | Periodic patterns | Repetitive cycles (Daily, Weekly, Yearly). |
| **Stationarity** | - | A series whose statistical properties (mean, variance) don't change over time. |

---

## 3. Model Selection Matrix

Use this table to decide which backend and model to use based on your constraints.

| Model Category | Example Model | Backend | Best For | Runtime | GPU Support | SOTA? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | `Naive`, `ETS` | sktime / Darts | Simple trends, debugging | ⚡ Very Fast | No | No |
| **Statistical** | `AutoARIMA` | sktime | Small datasets, high theory | 🟢 Fast | No | Classic |
| **Business-Grade** | `Prophet` | Prophet | Holidays, missing data, intuitive | 🟡 Medium | No | Robust |
| **Deep Learning** | `TiDE`, `N-BEATS` | Darts | Long horizons, non-linear patterns | 🔴 Slow | ✅ Yes | **SOTA** |
| **AutoML** | `AutoGluon` | AutoGluon | High accuracy without tuning | 🔴 Slow | ✅ Yes | **SOTA** |
| **Foundation** | `Chronos` | AutoGluon | Zero-shot forecasting | 🟡 Medium | ✅ Yes | **SOTA** |

---

## 4. When to Use Which?

### Scenarios

1.  **"I need a forecast in 5 seconds to show a trend."**
    *   **Backend**: `sktime` or `darts`
    *   **Model**: `NaiveSeasonal` or `ExponentialSmoothing`
2.  **"I have 500 different products and need accurate sales forecasts for each."**
    *   **Backend**: `autogluon`
    *   **Why**: Native Panel Data support. It learns patterns across different products simultaneously.
3.  **"My data is messy, has many missing values, and holidays are critical."**
    *   **Backend**: `prophet`
    *   **Why**: Prophet is extremely resilient to missing dates and handles additive/multiplicative seasonality well.
4.  **"I have a powerful GPU and want the absolute best performance for long-term forecasting."**
    *   **Backend**: `darts`
    *   **Model**: `TiDE` (Time-series Dense Encoder)
    *   **Why**: TiDE is highly optimized for GPU and handles long-term contexts better than RNNs.

---

## 5. Pro-Tips for GPU Training

*   **Memory Management**: Deep learning models (Darts, AutoGluon) scale with GPU RAM. If you hit `OutOfMemory` errors, reduce the `batch_size`.
*   **Arch compatibility**: Ensure your PyTorch version supports your GPU Architecture (e.g., RTX 50-series requires Torch Nightly as of early 2026).
*   **Panel Data**: GPU models shines most when training on "Panel Data" (many series) rather than just a single short series.

---

## 6. SOTA (State of the Art) Recommendations

If you want the best accuracy achievable today:
1.  **AutoGluon-TimeSeries**: Best automated ensemble approach.
2.  **TiDE (via Darts)**: Current top-tier architecture for multivariate/long-horizon forecasting.
3.  **Chronos (via AutoGluon)**: Best for "Zero-Shot" (predicting on data the model hasn't seen before).
