import useTelemetry from "../../hooks/useTelemetry";

export default function MetricsPanel() {
    const { emotion, workingMemory, answerMeta, somActivation } = useTelemetry();
    const activeNeurons = somActivation.filter(v => v > 0.05).length;

    return (
        <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8, background: "#fafafa" }}>
            <h5 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Live Telemetry</h5>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, fontSize: 13 }}>
                <div>Active Neurons: <strong>{activeNeurons} / 64</strong></div>
                <div>Arousal: <strong>{(emotion.arousal * 100).toFixed(0)}%</strong></div>
                <div>Valence: <strong>{(emotion.valence * 100).toFixed(0)}%</strong></div>
                <div>Novelty: <strong>{(emotion.novelty * 100).toFixed(0)}%</strong></div>
                <div>WM Subject: <strong>{workingMemory.subject}</strong></div>
                <div>WM Relation: <strong>{workingMemory.relation}</strong></div>
                <div>Status: <strong style={{ color: answerMeta.verified ? "#059669" : "#d97706" }}>{answerMeta.verified ? "VERIFIED ✓" : "UNVERIFIED"}</strong></div>
                <div>Kind: <strong>{answerMeta.kind}</strong></div>
            </div>
        </div>
    );
}
