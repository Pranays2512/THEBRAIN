export default function LogViewer() {
    return (
        <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8, background: "#fafafa", fontFamily: "monospace", fontSize: 12 }}>
            <h5 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888", fontFamily: "sans-serif" }}>Logs</h5>
            <div style={{ color: "#888" }}>[13:21:04] INFO: Session Started</div>
            <div style={{ color: "#888" }}>[13:21:05] INFO: Tool Invoked - web.search</div>
            <div style={{ color: "#cf222e" }}>[13:21:07] ERROR: Network Timeout</div>
        </div>
    );
}
