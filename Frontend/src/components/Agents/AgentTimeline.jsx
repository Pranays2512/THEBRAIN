import AgentStep from "./AgentStep";

export default function AgentTimeline({ plan = [], currentStep = 0 }) {
    return (
        <div style={{ padding: 18, background: "white", borderLeft: "1px solid rgba(0,0,0,0.08)", height: "100%", overflowY: "auto" }}>
            <h4 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Mission Timeline</h4>
            <hr style={{ border: "none", borderTop: "1px solid #eaeaea", marginBottom: 12 }} />
            
            {plan.length === 0 ? (
                <p style={{ color: "#888", fontSize: 14 }}>Waiting for plan...</p>
            ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {plan.map((step, index) => {
                        let status = "pending";
                        if (index < currentStep) status = "completed";
                        if (index === currentStep) status = "active";
                        
                        return <AgentStep key={step.id || index} step={step} status={status} />;
                    })}
                </div>
            )}
        </div>
    );
}
