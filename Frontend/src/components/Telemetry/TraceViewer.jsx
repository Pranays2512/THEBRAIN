export default function TraceViewer() {
    return (
        <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8, background: "#fafafa" }}>
            <h5 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Performance</h5>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 13 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}><span>LLM Request</span> <span>682 ms</span></div>
                <div style={{ display: "flex", justifyContent: "space-between" }}><span>GitHub MCP</span> <span>91 ms</span></div>
                <div style={{ display: "flex", justifyContent: "space-between" }}><span>Filesystem</span> <span>23 ms</span></div>
                <div style={{ display: "flex", justifyContent: "space-between" }}><span>Workflow</span> <span>913 ms</span></div>
                <hr style={{ border: "none", borderTop: "1px solid #eaeaea", margin: "4px 0" }} />
                <div style={{ display: "flex", justifyContent: "space-between", fontWeight: "bold" }}><span>Total</span> <span>1.82 s</span></div>
            </div>
        </div>
    );
}
