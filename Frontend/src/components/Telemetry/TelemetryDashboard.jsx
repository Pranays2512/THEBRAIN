import MetricsPanel from "./MetricsPanel";
import TraceViewer from "./TraceViewer";
import LogViewer from "./LogViewer";

export default function TelemetryDashboard() {
    return (
        <div style={{ padding: 18, background: "white", borderLeft: "1px solid rgba(0,0,0,0.08)", height: "100%", overflowY: "auto" }}>
            <h4 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Observability</h4>
            <hr style={{ border: "none", borderTop: "1px solid #eaeaea", marginBottom: 16 }} />
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <TraceViewer />
                <MetricsPanel />
                <LogViewer />
            </div>
        </div>
    );
}
