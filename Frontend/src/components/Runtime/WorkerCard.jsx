export default function WorkerCard({ worker }) {
    return (
        <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8, background: "#fafafa", marginBottom: 8 }}>
            <h5 style={{ margin: "0 0 4px 0", fontSize: 14 }}>
                {worker.status === "idle" ? "🟢" : worker.status === "running" ? "🟡" : "🔴"} {worker.id}
            </h5>
            <span style={{ fontSize: 12, color: "#888" }}>Status: {worker.status} | CPU: {worker.cpu}</span>
        </div>
    );
}
