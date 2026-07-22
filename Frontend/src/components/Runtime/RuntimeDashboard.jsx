import WorkerCard from "./WorkerCard";
import QueuePanel from "./QueuePanel";

export default function RuntimeDashboard() {
    const workers = [
        { id: "Local CPU", status: "running", cpu: 8 },
        { id: "GPU-1", status: "running", cpu: 32 },
        { id: "GPU-2", status: "idle", cpu: 32 },
        { id: "Cloud Worker", status: "disconnected", cpu: 0 }
    ];

    const queue = [
        { name: "Image Generation" },
        { name: "Repository Index" },
        { name: "Document OCR" }
    ];

    return (
        <div style={{ padding: 18, background: "white", borderLeft: "1px solid rgba(0,0,0,0.08)", height: "100%", overflowY: "auto" }}>
            <h4 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Workers</h4>
            <hr style={{ border: "none", borderTop: "1px solid #eaeaea", marginBottom: 12 }} />
            <div style={{ marginBottom: 20 }}>
                {workers.map(w => <WorkerCard key={w.id} worker={w} />)}
            </div>

            <QueuePanel queue={queue} />
            
            <div style={{ marginTop: 24, fontSize: 12, color: "#888" }}>
                <div>Workers Online: 3</div>
                <div>Jobs Queued: 3</div>
                <div>Worker Utilization: 66%</div>
            </div>
        </div>
    );
}
