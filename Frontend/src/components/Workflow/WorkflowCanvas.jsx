import WorkflowNodeUI from "./WorkflowNodeUI";

export default function WorkflowCanvas() {
    return (
        <div style={{ padding: 18, background: "white", borderLeft: "1px solid rgba(0,0,0,0.08)", height: "100%", overflowY: "auto" }}>
            <h4 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Workflow Canvas</h4>
            <hr style={{ border: "none", borderTop: "1px solid #eaeaea", marginBottom: 12 }} />
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {/* Real visualization elements go here based on graph */}
                <p style={{ color: "#888", fontSize: 14 }}>Live DAG visualization ready.</p>
            </div>
        </div>
    );
}
