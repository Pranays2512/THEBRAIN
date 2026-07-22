export default function QueuePanel({ queue = [] }) {
    return (
        <div>
            <h4 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Queue</h4>
            <hr style={{ border: "none", borderTop: "1px solid #eaeaea", marginBottom: 12 }} />
            {queue.length === 0 ? (
                <p style={{ color: "#888", fontSize: 14 }}>Queue is empty.</p>
            ) : (
                <ul style={{ padding: 0, listStyle: "none" }}>
                    {queue.map((job, idx) => (
                        <li key={idx} style={{ padding: 8, background: "#f5f5f5", marginBottom: 4, borderRadius: 6, fontSize: 14 }}>
                            {job.name || `Job #${idx}`}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
