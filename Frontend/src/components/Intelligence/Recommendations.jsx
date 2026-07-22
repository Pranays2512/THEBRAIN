import { CheckCircle2 } from "lucide-react";

export default function Recommendations() {
    const suggestions = [
        "Switch GitHub MCP endpoint",
        "Cache embeddings",
        "Use local model for summaries",
        "Parallelize documentation search"
    ];

    return (
        <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8, background: "#fafafa", marginBottom: 16 }}>
            <h5 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Optimization Suggestions</h5>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {suggestions.map((s, i) => (
                    <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 14 }}>
                        <CheckCircle2 size={16} color="#3db085" />
                        <span>{s}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
