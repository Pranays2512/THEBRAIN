import OptimizationPanel from "./OptimizationPanel";
import Recommendations from "./Recommendations";

export default function LearningDashboard() {
    return (
        <div style={{ padding: 18, background: "white", borderLeft: "1px solid rgba(0,0,0,0.08)", height: "100%", overflowY: "auto" }}>
            <h4 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Continuous Optimization</h4>
            <hr style={{ border: "none", borderTop: "1px solid #eaeaea", marginBottom: 16 }} />
            <Recommendations />
            <OptimizationPanel />
        </div>
    );
}
