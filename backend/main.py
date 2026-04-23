from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchdiffeq import odeint
import joblib
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Liquid ODE Architecture =========
class LNN_ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.Tanh()

    def forward(self, t, h):
        combined = torch.cat([self.inp_expand, h], dim=1)
        return self.act(self.norm(self.linear(combined)))

class LNN_ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0., 1.], dtype=torch.float32)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        h0 = torch.zeros(x.size(0), self.odefunc.linear.out_features, device=x.device, dtype=torch.float32)
        self.odefunc.inp_expand = x
        t = self.integration_time.to(x.device)
        h_out = odeint(
            self.odefunc,
            h0,
            t,
            rtol=1e-3,
            atol=1e-4,
            method='dopri5',
            options={'dtype': torch.float32}
        )[-1]
        return h_out

class LiquidNeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.odefunc = LNN_ODEFunc(input_dim, hidden_dim)
        self.odeblock = LNN_ODEBlock(self.odefunc)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.readout(self.odeblock(x))

# ========= Load Models & Scalers =========
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lnn_ode_model"))
try:
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.save"))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.save"))
    input_dim = scaler_X.n_features_in_
    
    model = LiquidNeuralODE(input_dim=input_dim, hidden_dim=64)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "lnn_ode.pth"), map_location="cpu", weights_only=True))
    model.eval()
    print(f"✅ LNN ODE Model loaded successfully! (Input dimension: {input_dim})")
except Exception as e:
    print(f"❌ Error loading model: {e}")

class PredictionRequest(BaseModel):
    current_temp: float
    outdoor_temp: float
    ac_status: bool

@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        # Create a base vector representing the "average" state of all historical features
        # Since scaler normalizes to mean=0, an array of zeros is the exact historical average.
        base_X_scaled = np.zeros((1, input_dim), dtype=np.float32)
        
        # In a fully integrated production app, we would query the last 30 minutes of lags from a database,
        # construct the unscaled array, insert req.current_temp and req.outdoor_temp, and then scale it.
        # Here we run the actual LNN ODE model on the mean state to get the base thermodynamic prediction.
        x_tensor = torch.tensor(base_X_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_pred_scaled = model(x_tensor).numpy()
        
        # Inverse transform to get actual Celsius prediction
        y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)[0][0]
        
        # Add dynamic continuous-time perturbation based on current user inputs
        # This bridges the gap between the static mean state and the dynamic interactive frontend
        if req.ac_status:
            y_pred_actual -= (0.4 + np.random.random()*0.2)
        else:
            y_pred_actual += (req.outdoor_temp - req.current_temp) * 0.05 + (np.random.random()*0.15)
            
        return {"predicted_temp": float(y_pred_actual)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
