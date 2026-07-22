import { CheckCircle2, Circle, Loader2 } from "lucide-react";

export default function AgentStep({ step, status }) {
    let icon;
    if (status === "active") {
        icon = <Loader2 size={16} style={{ animation: "spin 1s linear infinite", color: "#3db085" }} />;
    } else if (status === "completed") {
        icon = <CheckCircle2 size={16} color="#3db085" />;
    } else {
        icon = <Circle size={16} color="#888" />;
    }

    return (
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: 12, borderRadius: 12, border: "1px solid rgba(0,0,0,0.08)", background: status === "active" ? "#f0fdf4" : "#fafafa" }}>
            {icon}
            <div style={{ display: "flex", flexDirection: "column" }}>
                <span style={{ fontSize: 14, fontWeight: 500, color: status === "pending" ? "#888" : "#000" }}>{step.goal}</span>
                <span style={{ fontSize: 12, color: "#888" }}>Tool: {step.tool}</span>
            </div>
        </div>
    );
}
