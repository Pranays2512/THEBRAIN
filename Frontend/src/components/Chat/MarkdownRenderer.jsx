import React, { useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Code, ExternalLink } from "lucide-react";
import { useWorkspace } from "../../context/WorkspaceContext";
import CodeBlock from "./CodeBlock";
import "./MarkdownRenderer.css";

function ArtifactCard({ language, code }) {
    const { openArtifact } = useWorkspace();
    const lines = code.split("\n");
    const lineCount = lines.length;
    const title = `${language.toUpperCase()} Code Artifact (${lineCount} lines)`;

    const handleOpen = (e) => {
        if (e) {
            e.stopPropagation();
        }
        if (openArtifact) {
            openArtifact({
                id: title,
                title: title,
                type: "code",
                data: { language, code }
            });
        }
    };

    return (
        <div 
            className="artifact-chat-card material" 
            onClick={handleOpen}
            style={{
                margin: "14px 0",
                padding: "14px 18px",
                borderRadius: "20px",
                background: "var(--message-ai-bg, #ffffff)",
                border: "var(--message-border, 1px solid rgba(0,0,0,0.08))",
                boxShadow: "var(--message-shadow)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "12px",
                cursor: "pointer",
                transition: "all 0.2s ease"
            }}
        >
            <div style={{ display: "flex", alignItems: "center", gap: "12px", overflow: "hidden" }}>
                <div style={{
                    width: "40px",
                    height: "40px",
                    borderRadius: "14px",
                    background: "var(--button-mint-bg, rgba(154, 232, 199, 0.2))",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "var(--button-mint-text, #0f5d42)",
                    flexShrink: 0
                }}>
                    <Code size={20} />
                </div>
                <div style={{ overflow: "hidden" }}>
                    <div style={{ fontSize: "0.95rem", fontWeight: 700, color: "var(--text, #383634)", fontFamily: "'Shantell Sans', 'Patrick Hand', cursive", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {title}
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "var(--muted, #8C8782)", fontFamily: "'Patrick Hand', cursive", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        Click to view doodle code plan ({lineCount} lines)
                    </div>
                </div>
            </div>

            <button
                onClick={handleOpen}
                style={{
                    padding: "7px 14px",
                    borderRadius: "14px",
                    border: "var(--button-border, 1px solid rgba(0,0,0,0.06))",
                    background: "var(--button-bg, #ffffff)",
                    color: "var(--button-text, #2f2f2f)",
                    fontWeight: 700,
                    fontSize: "0.82rem",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    flexShrink: 0,
                    boxShadow: "var(--button-shadow)",
                    fontFamily: "'Patrick Hand', cursive"
                }}
            >
                <ExternalLink size={14} /> Open UI Panel
            </button>
        </div>
    );

}

export default function MarkdownRenderer({ children }) {
    return (
        <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
                p(props) {
                    return <p className="md-p" {...props} />;
                },

                h1(props) {
                    return <h1 className="md-h1" {...props} />;
                },

                h2(props) {
                    return <h2 className="md-h2" {...props} />;
                },

                h3(props) {
                    return <h3 className="md-h3" {...props} />;
                },

                ul(props) {
                    return <ul className="md-ul" {...props} />;
                },

                ol(props) {
                    return <ol className="md-ol" {...props} />;
                },

                li(props) {
                    return <li className="md-li" {...props} />;
                },

                blockquote(props) {
                    return (
                        <blockquote
                            className="md-quote"
                            {...props}
                        />
                    );
                },

                code({
                    inline,
                    className,
                    children
                }){
                    const language = className?.replace("language-", "") || "text";
                    const codeStr = String(children).replace(/\n$/, "");

                    if (inline) {
                        return (
                            <code className="inline-code">
                                {children}
                            </code>
                        );
                    }

                    // For multi-line code, create an Artifact Card & UI panel trigger!
                    if (codeStr.includes("\n") || codeStr.length > 50) {
                        return <ArtifactCard language={language} code={codeStr} />;
                    }

                    return (
                        <CodeBlock language={language}>
                            {codeStr}
                        </CodeBlock>
                    );
                }
            }}
        >
            {children}
        </ReactMarkdown>
    );
}
