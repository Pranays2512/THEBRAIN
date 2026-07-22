import { CheckCircle2, Circle, Loader2, XCircle } from "lucide-react";
import { TASK_STATUS } from "../../core/tasks/TaskTypes";

export default function TaskCard({ task }) {
    let icon;
    if (task.status === TASK_STATUS.RUNNING) {
        icon = <Loader2 size={16} style={{ animation: "spin 1s linear infinite", color: "#3db085" }} />;
    } else if (task.status === TASK_STATUS.COMPLETED) {
        icon = <CheckCircle2 size={16} color="#3db085" />;
    } else if (task.status === TASK_STATUS.FAILED) {
        icon = <XCircle size={16} color="#cf222e" />;
    } else {
        icon = <Circle size={16} color="#888" />;
    }

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 8, padding: 12, borderRadius: 12, border: "1px solid rgba(0,0,0,0.08)", marginBottom: 10, background: "#fafafa" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 14, fontWeight: 500 }}>
                {icon}
                <span>{task.title}</span>
            </div>
            {task.status === TASK_STATUS.RUNNING && (
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ flex: 1, height: 6, background: "rgba(0,0,0,0.05)", borderRadius: 3, overflow: "hidden" }}>
                        <div style={{ width: `${task.progress}%`, height: "100%", background: "#3db085", transition: "width 0.2s" }} />
                    </div>
                    <span style={{ fontSize: 12, color: "#888" }}>{task.progress}%</span>
                </div>
            )}
        </div>
    );
}
