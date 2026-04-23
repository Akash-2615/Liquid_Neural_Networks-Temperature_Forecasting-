# 🌊 Liquid Neural Network (LNN) ODE Model

> A dynamic, continuous-time Neural ODE architecture designed to model and predict real-world HVAC dynamics and indoor temperature variations.

## 📖 Overview

This project implements a **Liquid Neural Network (LNN)** utilizing Ordinary Differential Equations (ODEs) via PyTorch and `torchdiffeq`. I researched and chose Liquid Neural Networks for this task because **LNNs are uniquely capable of adapting to dynamic, continuous-time environments**, making them highly effective for forecasting physical systems like building thermodynamics where variables change fluidly over time.

The model predicts the **1-hour ahead indoor temperature** for a specific zone based on current and historical HVAC operational data, weather conditions, and time-based cyclical features.

## 🧠 Architecture: Liquid Neural ODE

The "Liquid" aspect of this model allows the network's hidden state to evolve continuously via an ODE solver, meaning the model can represent complex dynamic systems seamlessly. 

The architecture consists of:
1. **`LNN_ODEFunc`**: Defines the continuous-time dynamics of the hidden states, fusing current inputs with evolving hidden representations.
2. **`LNN_ODEBlock`**: Integrates the ODE function over time using the Dormand-Prince (`dopri5`) numerical solver.
3. **`LiquidNeuralODE`**: The overarching wrapper that processes the input, drives it through the continuous-time block, and generates the final temperature prediction via a linear readout layer.

## 📊 Dataset & Feature Engineering

The model trains on `floor2-zone1_with_weather.csv` which has been engineered to capture temporal dynamics:
- **Autoregressive Features:** Lags of indoor temperature (`temp_lag_1`, `temp_lag_2`) and short-term trends (`temp_roll_min_30`).
- **External Disturbances:** Outdoor `Temperature` and temperature differentials.
- **Control Variables:** AC power consumption (`z1_AC1(kW)`) and binary on/off states (`ac_on`).
- **Cyclical Time Features:** Sine/cosine embeddings of the hour, weekday, and weekend indicators.

## 🚀 Performance Metrics

The Liquid ODE model achieves exceptional predictive accuracy, proving its dominance in modeling thermodynamic inertia and dynamic environmental shifts:

- **✅ R² Score**: `0.9761`
- **📉 MAE**: `0.2240 °C`
- **📉 RMSE**: `0.4243 °C`
- **🎯 Accuracy (±0.5°C)**: `87.91%`

## 🏗️ Project Structure

```
LNN 2/
├── mn.ipynb                       # Main Jupyter Notebook containing the training & eval pipeline
├── output.png                     # Visualizations of predictions vs. ground truth
├── lnn_ode_model/
│   ├── lnn_ode.pth                # Trained PyTorch model weights
│   ├── scaler_X.save              # Fitted feature scaler
│   └── scaler_y.save              # Fitted target scaler
└── README.md                      # Project documentation
```

## ⚙️ Requirements

- `torch`
- `torchdiffeq` (for the ODE solver)
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib` / `seaborn` (for visualization)
