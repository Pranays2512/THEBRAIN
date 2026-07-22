export default function MetricsPanel() {
    return (
        <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8, background: "#fafafa" }}>
            <h5 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Live Metrics</h5>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, fontSize: 14 }}>
                <div>Sessions Active: <strong>14</strong></div>
                <div>Tasks Running: <strong>38</strong></div>
                <div>Workers Online: <strong>7</strong></div>
                <div>Plugins Loaded: <strong>16</strong></div>
                <div>Commands Executed: <strong>12,443</strong></div>
                <div>Avg Response: <strong>1.2s</strong></div>
            </div>
        </div>
    );
}
