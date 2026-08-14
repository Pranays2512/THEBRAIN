import MessageToolbar from "./MessageToolbar";
import BlockRenderer from "./Blocks/BlockRenderer";

import "./ChatMessage.css";

const InfinityAvatar = ({ isThinking }) => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path id="infinity-path" d="M12 12c-2-2.67-4-4-6-4a4 4 0 1 0 0 8c2 0 4-1.33 6-4Zm0 0c2 2.67 4 4 6 4a4 4 0 1 0 0-8c-2 0-4 1.33-6 4Z" opacity={isThinking ? 0.15 : 1} />
        {isThinking && (
            <circle r="2.5" fill="currentColor" stroke="none">
                <animateMotion dur="2s" repeatCount="indefinite">
                    <mpath href="#infinity-path" />
                </animateMotion>
            </circle>
        )}
    </svg>
);

export default function ChatMessage({
    role,
    blocks = [],
    status,
    createdAt,
    promptText = "",
    onResendMessage
}) {
    const msgStatus = status || "complete";
    const text = blocks.filter(b => b.type === "text").map(b => b.text).join("\n");
    const canShowToolbar = msgStatus === "complete" || msgStatus === "error" || msgStatus === "cancelled";

    return (
        <div className={`message-row ${role}`}>
            <div className="message-avatar">
                {role === "assistant" ? <InfinityAvatar isThinking={msgStatus === "streaming"} /> : "You"}
            </div>
            <div className="message-stack">
                <div className={`message-card material ${role}`}>
                    <BlockRenderer blocks={blocks} />
                    
                    {msgStatus === "streaming" && (
                        <span className="typing-cursor" />
                    )}
                    {msgStatus === "error" && (
                        <div className="error-badge">
                            Generation failed
                        </div>
                    )}
                    {msgStatus === "cancelled" && (
                        <div className="cancelled-badge">
                            Stopped
                        </div>
                    )}

                    <div className="message-time">
                        {
                            createdAt
                            ? new Date(createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                            : new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                        }
                    </div>
                </div>

                {canShowToolbar && (
                    <MessageToolbar
                        onCopy={() => {
                            navigator.clipboard.writeText(text);
                        }}
                        onRetry={() => {
                            if (promptText && onResendMessage) {
                                onResendMessage(promptText);
                            }
                        }}
                        retryDisabled={!promptText || !onResendMessage}
                    />
                )}
            </div>
        </div>
    );
}
