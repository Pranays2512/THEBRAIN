import { useMCP } from "../../context/MCPContext";
import MCPRegistry from "../../core/mcp/MCPRegistry";
import { useState } from "react";

export default function ToolExplorer() {
    const [tools] = useState(() => MCPRegistry.getTools());

    return (
        <div style={{ padding: 18, background: "white", borderLeft: "1px solid rgba(0,0,0,0.08)", height: "100%", overflowY: "auto" }}>
            <h4 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Tools</h4>
            <hr style={{ border: "none", borderTop: "1px solid #eaeaea", marginBottom: 12 }} />
            {tools.length === 0 ? (
                <p style={{ color: "#888", fontSize: 14 }}>No tools discovered yet.</p>
            ) : (
                <ul style={{ listStyle: "none", padding: 0 }}>
                    {tools.map(t => (
                        <li key={t.name} style={{ padding: 8, background: "#f5f5f5", marginBottom: 4, borderRadius: 6, fontSize: 14 }}>
                            {t.name}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
