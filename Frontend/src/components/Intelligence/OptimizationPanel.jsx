export default function OptimizationPanel() {
    return (
        <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8, background: "#fafafa" }}>
            <h5 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Estimated Improvement</h5>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 14 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}><span>Latency</span> <span style={{ color: "#3db085", fontWeight: "bold" }}>-28%</span></div>
                <div style={{ display: "flex", justifyContent: "space-between" }}><span>Cost</span> <span style={{ color: "#3db085", fontWeight: "bold" }}>-19%</span></div>
                <div style={{ display: "flex", justifyContent: "space-between" }}><span>Token Usage</span> <span style={{ color: "#3db085", fontWeight: "bold" }}>-14%</span></div>
            </div>
        </div>
    );
}
