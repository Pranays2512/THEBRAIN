import React from 'react';
import BrainMatrix from "./BrainMatrix";
import DoodleGraph from "./DoodleGraph";
import useTelemetry from "../../hooks/useTelemetry";
import "./Dashboard.css";

export default function Dashboard() {
    const { 
        arousalHistory, 
        valenceHistory, 
        noveltyHistory, 
        surpriseHistory,
        workingMemory,
        answerMeta
    } = useTelemetry();

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <div>
                    <h2>Brain Telemetry & Monitoring</h2>
                    <p style={{ margin: 0, fontSize: "0.85rem", opacity: 0.7 }}>
                        Active Concept: <strong>{workingMemory.subject}</strong> → <strong>{workingMemory.relation}</strong> → <strong>{workingMemory.object}</strong>
                    </p>
                </div>
                <span className="live-pill" style={{
                    fontSize: "0.75rem",
                    fontWeight: "bold",
                    padding: "4px 10px",
                    borderRadius: "12px",
                    background: answerMeta.verified ? "#8BD4BA" : "#fbbf24",
                    color: "#1a1a1a"
                }}>
                    {answerMeta.verified ? "VERIFIED ✓" : `KIND: ${answerMeta.kind}`}
                </span>
            </header>
            
            <div className="dashboard-grid">
                <div className="dashboard-card matrix-card material">
                    <h3 className="card-title">SOM Memory Matrix (8x8)</h3>
                    <div className="matrix-wrapper">
                        <BrainMatrix />
                    </div>
                </div>

                <div className="dashboard-card material">
                    <DoodleGraph 
                        data={arousalHistory.map(v => Math.round(v * 100))} 
                        label="Cognitive Arousal" 
                        detail="Live neural activation intensity."
                        color="var(--mint)" 
                        unit="%"
                    />
                </div>
                <div className="dashboard-card material">
                    <DoodleGraph 
                        data={valenceHistory.map(v => Math.round(v * 100))} 
                        label="Confidence / Valence" 
                        detail="Brain's internal answer certainty."
                        color="#3b82f6" 
                        unit="%"
                    />
                </div>
                <div className="dashboard-card material">
                    <DoodleGraph 
                        data={noveltyHistory.map(v => Math.round(v * 100))} 
                        label="Input Novelty" 
                        detail="Uniqueness of incoming phrase."
                        color="#ec4899" 
                        unit="%"
                    />
                </div>
                <div className="dashboard-card material">
                    <DoodleGraph 
                        data={surpriseHistory.map(v => Math.round(v * 100))} 
                        label="Prediction Surprise" 
                        detail="Predictive coding error signal."
                        color="#eab308" 
                        unit="%"
                    />
                </div>
            </div>
        </div>
    );
}
