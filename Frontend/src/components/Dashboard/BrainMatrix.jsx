import useTelemetry from "../../hooks/useTelemetry";
import { motion } from "framer-motion";

export default function BrainMatrix() {
    const { somActivation } = useTelemetry();

    return (
        <div style={{ width: "100%", height: "100%", padding: "12px", display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: "6px" }}>
            {somActivation.map((val, i) => {
                const isActive = val > 0.05;
                const opacity = Math.min(val * 0.9 + 0.15, 1.0);
                return (
                    <motion.div
                        key={i}
                        animate={{
                            scale: isActive ? 1.05 : 1.0,
                            opacity: opacity,
                        }}
                        transition={{ duration: 0.3 }}
                        style={{
                            aspectRatio: "1",
                            borderRadius: "6px",
                            backgroundColor: isActive ? "var(--mint)" : "rgba(160, 165, 176, 0.2)",
                            boxShadow: isActive ? "0 0 10px rgba(139, 212, 186, 0.8)" : "none",
                            border: isActive ? "1px solid var(--mint)" : "1px solid rgba(255,255,255,0.05)"
                        }}
                        title={`Neuron ${i}: ${(val * 100).toFixed(0)}%`}
                    />
                );
            })}
        </div>
    );
}
