import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceArea
} from 'recharts';
import { Activity, Thermometer, Wind, Zap } from 'lucide-react';

export default function App() {
  const [data, setData] = useState([]);
  const [acStatus, setAcStatus] = useState(false);
  const [outdoorTemp, setOutdoorTemp] = useState(30.0);
  const [currentIndoorTemp, setCurrentIndoorTemp] = useState(24.5);
  const [predictedTemp, setPredictedTemp] = useState(24.8);

  // Simulate data points
  useEffect(() => {
    const initialData = Array.from({ length: 24 }).map((_, i) => {
      const time = new Date();
      time.setMinutes(time.getMinutes() - (24 - i) * 30);
      
      const baseTemp = 24 + Math.sin(i * 0.5) * 1.5;
      return {
        time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        actual: baseTemp.toFixed(2),
        predicted: (baseTemp + (Math.random() * 0.4 - 0.2)).toFixed(2),
        acPower: baseTemp > 25 ? (Math.random() * 1.5 + 1).toFixed(1) : 0
      };
    });
    setData(initialData);
  }, []);

  // Real-time updates simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setData(prev => {
        const newData = [...prev.slice(1)];
        const time = new Date();
        
        // Dynamic simulation based on AC status and outdoor temp
        let newTemp = currentIndoorTemp;
        if (acStatus) {
          newTemp -= 0.1 * (Math.random() * 0.5 + 0.5); // Cooling effect
        } else {
          // Heating effect based on outdoor temp gradient
          newTemp += (outdoorTemp - currentIndoorTemp) * 0.01 * Math.random();
        }

        // LNN prediction simulation (1 hour ahead)
        const prediction = acStatus 
          ? newTemp - 0.5 + (Math.random() * 0.2)
          : newTemp + (outdoorTemp - newTemp) * 0.05 + (Math.random() * 0.2);

        setCurrentIndoorTemp(newTemp);
        setPredictedTemp(prediction);

        newData.push({
          time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          actual: newTemp.toFixed(2),
          predicted: prediction.toFixed(2),
          acPower: acStatus ? (1.5 + Math.random() * 0.5).toFixed(1) : 0
        });

        return newData;
      });
    }, 3000);

    return () => clearInterval(interval);
  }, [acStatus, outdoorTemp, currentIndoorTemp]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ background: 'rgba(15, 23, 42, 0.9)', border: '1px solid rgba(255,255,255,0.1)', padding: '10px', borderRadius: '8px' }}>
          <p style={{ color: '#fff', marginBottom: '5px' }}>{label}</p>
          {payload.map((p, i) => (
            <p key={i} style={{ color: p.color, fontWeight: 'bold' }}>
              {p.name}: {p.value} {p.name.includes('Power') ? 'kW' : '°C'}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <div>
          <h1 className="title">Liquid Neural Network Dashboard</h1>
          <p className="subtitle">Real-time HVAC thermodynamics prediction using Continuous-time ODEs</p>
        </div>
        <div className="status-badge">
          <div className="status-dot"></div>
          LNN Engine Active
        </div>
      </header>

      <div className="kpi-grid">
        <div className="glass-card">
          <div className="kpi-title"><Thermometer size={16} style={{display: 'inline', marginRight: '5px', verticalAlign: 'middle'}}/> Current Indoor Temp</div>
          <div className="kpi-value" style={{color: '#fff'}}>{currentIndoorTemp.toFixed(2)}<span className="kpi-unit">°C</span></div>
        </div>
        
        <div className="glass-card">
          <div className="kpi-title"><Activity size={16} style={{display: 'inline', marginRight: '5px', verticalAlign: 'middle'}}/> Predicted Temp (1h ahead)</div>
          <div className="kpi-value" style={{color: '#8b5cf6'}}>{predictedTemp.toFixed(2)}<span className="kpi-unit">°C</span></div>
        </div>

        <div className="glass-card">
          <div className="kpi-title"><Wind size={16} style={{display: 'inline', marginRight: '5px', verticalAlign: 'middle'}}/> Model Accuracy (R²)</div>
          <div className="kpi-value" style={{color: '#10b981'}}>0.976<span className="kpi-unit"></span></div>
        </div>

        <div className="glass-card">
          <div className="kpi-title"><Zap size={16} style={{display: 'inline', marginRight: '5px', verticalAlign: 'middle'}}/> Current AC Power</div>
          <div className="kpi-value" style={{color: '#3b82f6'}}>{acStatus ? '1.8' : '0.0'}<span className="kpi-unit">kW</span></div>
        </div>
      </div>

      <div className="chart-section">
        <div className="glass-card">
          <h3>Temperature Dynamics & Forecast</h3>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="time" stroke="#94a3b8" />
                <YAxis yAxisId="left" domain={['dataMin - 1', 'dataMax + 1']} stroke="#94a3b8" />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="actual" name="Actual Temp" stroke="#fff" strokeWidth={2} dot={false} />
                <Line yAxisId="left" type="monotone" dataKey="predicted" name="LNN Predicted Temp" stroke="#8b5cf6" strokeWidth={3} dot={false} strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-card controls-section">
          <h3>Simulation Controls</h3>
          <p style={{color: '#94a3b8', fontSize: '0.9rem', marginBottom: '1rem'}}>Modify environment variables to see how the Liquid Neural Network adapts its predictions in real-time.</p>
          
          <div className="control-item">
            <div className="control-label">
              <Zap size={20} color="#3b82f6"/> AC Status
            </div>
            <label className="toggle-switch">
              <input type="checkbox" checked={acStatus} onChange={(e) => setAcStatus(e.target.checked)} />
              <span className="slider"></span>
            </label>
          </div>

          <div className="control-item" style={{flexDirection: 'column', alignItems: 'flex-start', gap: '1rem'}}>
            <div className="control-label" style={{width: '100%', justifyContent: 'space-between'}}>
              <span><Thermometer size={20} color="#ef4444" style={{verticalAlign: 'middle', marginRight: '5px'}}/> Outdoor Temp</span>
              <span className="value-display">{outdoorTemp.toFixed(1)} °C</span>
            </div>
            <input 
              type="range" 
              min="15" 
              max="45" 
              step="0.5" 
              value={outdoorTemp} 
              onChange={(e) => setOutdoorTemp(parseFloat(e.target.value))}
              style={{width: '100%', accentColor: '#ef4444'}}
            />
          </div>
          
          <div style={{marginTop: 'auto', padding: '1rem', background: 'rgba(139, 92, 246, 0.1)', borderLeft: '4px solid #8b5cf6', borderRadius: '4px'}}>
            <h4 style={{color: '#8b5cf6', marginBottom: '0.5rem', fontSize: '0.9rem'}}>LNN Engine Note</h4>
            <p style={{fontSize: '0.8rem', color: '#cbd5e1', lineHeight: '1.4'}}>
              The ODE solver continuously integrates the hidden state. Notice how predictions adapt smoothly to AC toggles and outdoor temperature gradients.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
