export default function WorkflowNodeUI({ node }) {
    return (
        <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8, background: "#fafafa" }}>
            <h5 style={{ margin: "0 0 4px 0", fontSize: 14 }}>{node.tool}</h5>
            <span style={{ fontSize: 12, color: "#888" }}>Status: {node.status}</span>
        </div>
    );
}
