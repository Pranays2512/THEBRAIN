export default function EventTimeline({ events = [] }) {
    return (
        <div style={{ padding: 18, background: "white", borderLeft: "1px solid rgba(0,0,0,0.08)", height: "100%", overflowY: "auto" }}>
            <h4 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Audit Trail</h4>
            <hr style={{ border: "none", borderTop: "1px solid #eaeaea", marginBottom: 12 }} />
            {events.length === 0 ? (
                <p style={{ color: "#888", fontSize: 14 }}>No events recorded.</p>
            ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {events.map((e, i) => (
                        <div key={e.id || i} style={{ display: "flex", gap: 8, fontSize: 13 }}>
                            <span style={{ color: "#888", width: 45 }}>{new Date(e.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                            <span style={{ fontWeight: 500 }}>{e.type}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
