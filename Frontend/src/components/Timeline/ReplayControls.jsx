import { Play, Pause, SkipBack, SkipForward } from "lucide-react";

export default function ReplayControls() {
    return (
        <div style={{ display: "flex", alignItems: "center", gap: 12, padding: 12, background: "#fafafa", borderRadius: 8, border: "1px solid #eaeaea" }}>
            <button style={{ background: "transparent", border: "none", cursor: "pointer", display: "flex", color: "#555" }}><SkipBack size={18} /></button>
            <button style={{ background: "transparent", border: "none", cursor: "pointer", display: "flex", color: "#555" }}><Play size={18} /></button>
            <button style={{ background: "transparent", border: "none", cursor: "pointer", display: "flex", color: "#555" }}><Pause size={18} /></button>
            <button style={{ background: "transparent", border: "none", cursor: "pointer", display: "flex", color: "#555" }}><SkipForward size={18} /></button>
            <div style={{ flex: 1, height: 4, background: "#ddd", borderRadius: 2, position: "relative" }}>
                <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: "40%", background: "#3db085", borderRadius: 2 }} />
                <div style={{ position: "absolute", left: "40%", top: -4, width: 12, height: 12, background: "white", border: "2px solid #3db085", borderRadius: "50%", transform: "translateX(-50%)", cursor: "pointer" }} />
            </div>
        </div>
    );
}
