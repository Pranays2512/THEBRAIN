import React from 'react';
import BrainMatrix from "./BrainMatrix";
import DoodleGraph from "./DoodleGraph";
import "./Dashboard.css";

const MOCK_DATA_TRAINING = [0.85, 0.72, 0.65, 0.58, 0.42, 0.38, 0.31, 0.25, 0.18, 0.15];
const MOCK_DATA_MEMORY = [10, 25, 30, 45, 55, 60, 68, 75, 82, 88];
const MOCK_DATA_COGNITIVE = [20, 35, 30, 50, 45, 60, 80, 75, 90, 85];
const MOCK_DATA_REASONING = [10, 15, 12, 20, 25, 40, 30, 50, 45, 65];

export default function Dashboard() {
    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <h2>Brain Monitoring</h2>
                <p>Live metrics and matrix state.</p>
            </header>
            
            <div className="dashboard-grid">
                <div className="dashboard-card matrix-card material">
                    <h3 className="card-title">Neural Matrix</h3>
                    <div className="matrix-wrapper">
                        <BrainMatrix />
                    </div>
                </div>

                <div className="dashboard-card material">
                    <DoodleGraph 
                        data={MOCK_DATA_TRAINING} 
                        label="Training Loss" 
                        detail="Model convergence over last 10 steps."
                        color="#ef4444" 
                        unit=""
                    />
                </div>
                <div className="dashboard-card material">
                    <DoodleGraph 
                        data={MOCK_DATA_MEMORY} 
                        label="Retention Memory" 
                        detail="Stored user context and preferences."
                        color="var(--mint)" 
                        unit="%"
                    />
                </div>
                <div className="dashboard-card material">
                    <DoodleGraph 
                        data={MOCK_DATA_COGNITIVE} 
                        label="Cognitive Load" 
                        detail="Compute intensity of current operations."
                        color="var(--text)" 
                        unit="%"
                    />
                </div>
                <div className="dashboard-card material">
                    <DoodleGraph 
                        data={MOCK_DATA_REASONING} 
                        label="Reasoning Depth" 
                        detail="Logic chain length and complexity."
                        color="#eab308" 
                        unit=" nodes"
                    />
                </div>
            </div>
        </div>
    );
}
