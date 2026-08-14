import { Check, Copy, X } from "lucide-react";
import { useState } from "react";
import { useWorkspace } from "../../context/WorkspaceContext";
import ArtifactRenderer from "../Artifacts/ArtifactRenderer";

export default function ArtifactPanel() {
    const { artifact, closeArtifact } = useWorkspace();
    const [copied, setCopied] = useState(false);

    if (!artifact) return null;

    const handleCopy = async (e) => {
        e.preventDefault();
        e.stopPropagation();

        let text = "";
        if (artifact.data?.code) text = artifact.data.code;
        else if (artifact.data?.markdown) text = artifact.data.markdown;
        else if (typeof artifact.data === "string") text = artifact.data;

        if (!text) return;

        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
    };

    return (
        <div
            className="artifact-panel-container"
            style={{
                height: "100%",
                width: "100%",
                padding: "8px",
                boxSizing: "border-box"
            }}
        >
            <div
                className="message-card material ai artifact-bubble"
                style={{
                    height: "100%",
                    width: "100%",
                    maxWidth: "none",
                    minWidth: 0,
                    display: "flex",
                    flexDirection: "column",
                    padding: 0,
                    borderRadius: "var(--radius-lg, 28px)",
                    background: "var(--message-ai-bg, #ffffff)",
                    boxShadow: "var(--message-shadow)",
                    border: "var(--message-border, 1px solid rgba(0,0,0,0.06))",
                    overflow: "hidden",
                    color: "var(--text, #383634)",
                    boxSizing: "border-box"
                }}
            >
                <div
                    className="artifact-header"
                    style={{
                        display: "flex",
                        justifyContent: "flex-end",
                        alignItems: "center",
                        gap: 8,
                        padding: "8px 10px",
                        borderBottom: "1px solid rgba(0,0,0,0.05)"
                    }}
                >
                    <button
                        onClick={handleCopy}
                        style={{
                            width: "32px",
                            height: "32px",
                            cursor: "pointer",
                            borderRadius: "50%",
                            border: "var(--button-border, 1px solid rgba(0,0,0,0.06))",
                            background: "var(--button-bg, #ffffff)",
                            color: copied ? "#0f5d42" : "var(--button-text, #2f2f2f)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            boxShadow: "var(--button-shadow)"
                        }}
                        title={copied ? "Copied" : "Copy artifact"}
                    >
                        {copied ? <Check size={15} /> : <Copy size={15} />}
                    </button>

                    <button
                        onClick={closeArtifact}
                        style={{
                            width: "32px",
                            height: "32px",
                            cursor: "pointer",
                            borderRadius: "50%",
                            border: "none",
                            background: "rgba(0,0,0,0.05)",
                            color: "var(--muted, #8C8782)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center"
                        }}
                        title="Close artifact"
                    >
                        <X size={17} />
                    </button>
                </div>

                <div
                    className="artifact-content"
                    style={{
                        flex: 1,
                        minHeight: 0,
                        overflow: "auto",
                        padding: "4px 8px 10px",
                        boxSizing: "border-box"
                    }}
                >
                    <ArtifactRenderer artifact={artifact} />
                </div>
            </div>
        </div>
    );
}
