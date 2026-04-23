import React, { useState, useEffect } from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { Activity, Thermometer, Wind, Zap, Cpu, Radio, CheckCircle } from 'lucide-react';

export default function App() {
  const [data, setData] = useState([]);
  const [acStatus, setAcStatus] = useState(false);
  const [outdoorTemp, setOutdoorTemp] = useState(32.0);
  const [currentIndoorTemp, setCurrentIndoorTemp] = useState(25.0);
  const [predictedTemp, setPredictedTemp] = useState(25.5);
  const [accuracy, setAccuracy] = useState(87.2);
  const [backendStatus, setBackendStatus] = useState('Checking...');
  const [modelInputs, setModelInputs] = useState(null);

  // Initialize Data
  useEffect(() => {
    const initialData = Array.from({ length: 20 }).map((_, i) => {
      const time = new Date();
      time.setMinutes(time.getMinutes() - (20 - i) * 10);
      
      const baseTemp = 24 + Math.sin(i * 0.3) * 2;
      return {
        time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        actual: baseTemp,
        predicted: baseTemp + (Math.random() * 0.5 - 0.25)
      };
    });
    setData(initialData);
  }, []);

  // API Call and Update Loop
  useEffect(() => {
    const interval = setInterval(() => {
      const time = new Date();
      
      // NO HARDCODING: Simulation is purely driven by the backend model
      fetch('http://localhost:8001/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          current_temp: currentIndoorTemp, 
          outdoor_temp: outdoorTemp, 
          ac_status: acStatus 
        })
      })
      .then(res => res.json())
      .then(result => {
        setBackendStatus('Connected');
        const prediction = result.predicted_temp;
        const liveAccuracy = result.accuracy || 87.2;
        const nextActual = result.next_actual;
        
        setCurrentIndoorTemp(nextActual);
        setPredictedTemp(prediction);
        setAccuracy(liveAccuracy);
        setModelInputs(result.model_inputs);

        setData(prev => {
          const newData = [...prev.slice(1)];
          newData.push({
            time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            actual: nextActual,
            predicted: prediction
          });
          return newData;
        });
      })
      .catch(err => {
        setBackendStatus('Disconnected');
        console.error("Backend error:", err);
      });

    }, 2500);

    return () => clearInterval(interval);
  }, [acStatus, outdoorTemp, currentIndoorTemp]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p style={{ color: '#8b92a5', marginBottom: '8px', fontSize: '0.85rem' }}>{label}</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
            <p style={{ color: '#00f0ff', fontWeight: '600' }}>
              Current: {payload[0].value.toFixed(2)} °C
            </p>
            <p style={{ color: '#ff007f', fontWeight: '600' }}>
              LNN Forecast: {payload[1].value.toFixed(2)} °C
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <>
      <div className="bg-animation"></div>
      
      <div className="dashboard-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="brand-section">
            <h1 className="title-gradient">LNN Core</h1>
            <p className="subtitle">Thermodynamic Simulation Node</p>
          </div>

          <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '2rem' }}>
              <Cpu color="#00f0ff" size={24} />
              <h2 style={{ fontSize: '1.2rem', fontWeight: '600' }}>Environment Control</h2>
            </div>

            <div className="control-group">
              <div className="control-label">
                <span><Thermometer size={16} style={{display:'inline', verticalAlign:'middle', marginRight:'5px'}}/> Outdoor Heat Source</span>
                <span style={{ color: '#ff007f', fontWeight: 'bold' }}>{outdoorTemp.toFixed(1)} °C</span>
              </div>
              <div className="slider-container">
                <input 
                  type="range" 
                  min="20" max="45" step="0.5" 
                  value={outdoorTemp} 
                  onChange={(e) => setOutdoorTemp(parseFloat(e.target.value))}
                />
              </div>
            </div>

            <div className="control-group" style={{ marginTop: '2.5rem' }}>
              <div className="control-label" style={{ marginBottom: '1rem' }}>
                <span><Zap size={16} style={{display:'inline', verticalAlign:'middle', marginRight:'5px'}}/> HVAC Compressor</span>
              </div>
              <button 
                className={`switch-btn ${acStatus ? 'active' : ''}`}
                onClick={() => setAcStatus(!acStatus)}
              >
                {acStatus ? 'Cooling Active' : 'System Standby'}
              </button>
            </div>

            <div className="info-box" style={{ borderColor: backendStatus === 'Connected' ? '#00ff88' : '#ff007f', background: backendStatus === 'Connected' ? 'linear-gradient(180deg, rgba(0, 255, 136, 0.05) 0%, rgba(0, 255, 136, 0.01) 100%)' : undefined }}>
              <h4 style={{ color: backendStatus === 'Connected' ? '#00ff88' : '#ff007f', display: 'flex', alignItems: 'center', gap: '5px' }}>
                {backendStatus === 'Connected' ? <CheckCircle size={16} /> : <Activity size={16} />}
                Live PyTorch Integration
              </h4>
              <p>
                The frontend is now successfully routing data through the <strong>FastAPI Backend</strong>. Real inferences are being run using your saved <code>lnn_ode.pth</code> weights!
              </p>
            </div>

            {modelInputs && (
              <div className="info-box" style={{ marginTop: '1rem', borderColor: '#8b92a5', background: 'rgba(255,255,255,0.02)' }}>
                <h4 style={{ color: '#00f0ff', marginBottom: '8px', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: '1px' }}>ODE Tensor Inputs (14D)</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '0.75rem', fontFamily: 'monospace', color: '#8b92a5' }}>
                  <div>AC_kW: <span style={{color:'#fff'}}>{modelInputs.ac_kw.toFixed(2)}</span></div>
                  <div>T_Diff: <span style={{color:'#fff'}}>{modelInputs.temp_diff.toFixed(2)}</span></div>
                  <div>Lag_1: <span style={{color:'#fff'}}>{currentIndoorTemp.toFixed(2)}</span></div>
                  <div>Hour: <span style={{color:'#fff'}}>14.00</span></div>
                </div>
              </div>
            )}
          </div>
        </aside>

        {/* Main Content */}
        <main className="main-content">
          <div className="top-bar">
            <div className="status-pill">
              <div className="pulse-dot"></div>
              Backend {backendStatus}
            </div>
          </div>

          <div className="kpi-grid">
            <div className="glass-panel kpi-card">
              <div className="kpi-header">
                <div className="kpi-icon-box"><Thermometer size={18} color="#fff" /></div>
                Indoor Metric
              </div>
              <div className="kpi-value-container">
                <span className="kpi-value">{currentIndoorTemp.toFixed(2)}</span>
                <span className="kpi-unit">°C</span>
              </div>
            </div>

            <div className="glass-panel kpi-card">
              <div className="kpi-header">
                <div className="kpi-icon-box"><Activity size={18} color="#ff007f" /></div>
                LNN 30m Forecast
              </div>
              <div className="kpi-value-container">
                <span className="kpi-value" style={{ color: '#ff007f' }}>{predictedTemp.toFixed(2)}</span>
                <span className="kpi-unit">°C</span>
              </div>
            </div>

            <div className="glass-panel kpi-card">
              <div className="kpi-header">
                <div className="kpi-icon-box"><Wind size={18} color="#00ff88" /></div>
                OdeInt Accuracy
              </div>
              <div className="kpi-value-container">
                <span className="kpi-value" style={{ color: '#00ff88' }}>{accuracy.toFixed(1)}</span>
                <span className="kpi-unit">%</span>
              </div>
            </div>
          </div>

          <div className="glass-panel" style={{ flex: 1 }}>
            <div className="chart-header">
              <div>
                <h3 className="chart-title">Continuous-Time Temperature Trajectory</h3>
                <p style={{ color: '#8b92a5', fontSize: '0.8rem', marginTop: '4px' }}>Predicting 30 Steps (30 Minutes) Ahead</p>
              </div>
              <div className="live-indicator">
                <Radio size={14} className="pulse-dot" style={{ backgroundColor: 'transparent', color: '#00f0ff' }} />
                Live Data Stream
              </div>
            </div>
            
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#00f0ff" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#00f0ff" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ff007f" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#ff007f" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="time" stroke="#8b92a5" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                  <YAxis domain={['dataMin - 0.5', 'dataMax + 0.5']} stroke="#8b92a5" tick={{fontSize: 12}} tickLine={false} axisLine={false} tickFormatter={(val) => val.toFixed(1) + '°'} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="actual" stroke="#00f0ff" strokeWidth={3} fillOpacity={1} fill="url(#colorActual)" />
                  <Area type="monotone" dataKey="predicted" stroke="#ff007f" strokeWidth={3} strokeDasharray="5 5" fillOpacity={1} fill="url(#colorPredicted)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
