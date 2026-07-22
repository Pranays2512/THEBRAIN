import { useEffect, useRef, useState } from "react";
import { Paperclip, Send, Mic, Sparkles, Square, ChevronDown, ChevronUp } from "lucide-react";
import { ClayInput, ClayIconButton } from "../Common";
import useAttachments from "../../hooks/useAttachments";
import AttachmentPreview from "./AttachmentPreview";
import "./AttachmentPreview.css";
import "./Composer.css";

export default function Composer({
    onSend = () => {},
    streaming,
    onStop
}) {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [message, setMessage] = useState("");
    const textareaRef = useRef(null);
    const inputRef = useRef(null);

    const {
        attachments,
        addFiles,
        removeAttachment,
        clear
    } = useAttachments();

    function send() {
        const text = message.trim();
        if (!text && attachments.length === 0) return;
        
        onSend({ text, attachments });
        
        setMessage("");
        clear();
        
        requestAnimationFrame(() => {
            if (textareaRef.current) {
                textareaRef.current.style.height = "26px";
            }
        });
    }

    useEffect(() => {
        if (!textareaRef.current) return;
        requestAnimationFrame(() => {
            if (!textareaRef.current) return;
            if (message === "") {
                textareaRef.current.style.height = "";
                return;
            }
            textareaRef.current.style.height = "26px";
            textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
        });
    }, [message]);

    function keyDown(e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    }

    return (
        <>
            <div 
                className={`composer-shell material ${isCollapsed ? 'collapsed' : ''}`}
                onDragOver={e => { e.preventDefault(); }}
                onDrop={e => {
                    e.preventDefault();
                    addFiles(e.dataTransfer.files);
                }}
                style={{ display: "flex", flexDirection: "column", alignItems: "stretch", gap: 0 }}
            >
            <input
                ref={inputRef}
                type="file"
                multiple
                hidden
                accept="image/*,.pdf,.txt"
                onChange={e => addFiles(e.target.files)}
            />

            {attachments.length > 0 && (
                <div className="attachment-grid">
                    {attachments.map(file => (
                        <AttachmentPreview
                            key={file.id}
                            attachment={file}
                            onRemove={() => removeAttachment(file.id)}
                        />
                    ))}
                </div>
            )}

            <div className="composer-row" style={{ display: "flex", alignItems: "flex-end", gap: "18px", width: "100%" }}>
                <div className="composer-left">
                    <ClayIconButton
                        size="sm"
                        icon={<Paperclip />}
                        onClick={() => { inputRef.current.click(); }}
                    />
                </div>
                <div className="composer-center">
                    <ClayInput
                        multiline
                        rows={1}
                        ref={textareaRef}
                        value={message}
                        placeholder="Ask Brain AI anything..."
                        onKeyDown={keyDown}
                        onChange={(e) => setMessage(e.target.value)}
                        onPaste={e => {
                            const files = [...e.clipboardData.files];
                            if (files.length) addFiles(files);
                        }}
                    />
                </div>
                <div className="composer-right">
                    <ClayIconButton
                        size="sm"
                        variant="mint"
                        icon={<Sparkles />}
                    />
                    {streaming ? (
                        <ClayIconButton
                            size="sm"
                            variant="dark"
                            icon={<Square />}
                            onClick={onStop}
                            className="send-button"
                        />
                    ) : (message.trim() || attachments.length > 0) ? (
                        <ClayIconButton
                            size="sm"
                            variant="dark"
                            icon={<Send />}
                            onClick={send}
                            className="send-button"
                        />
                    ) : (
                        <ClayIconButton
                            size="sm"
                            icon={<Mic />}
                            className="mic-button"
                        />
                    )}
                    <div style={{ width: '1px', height: '24px', background: 'var(--muted)', opacity: 0.3, margin: '0 4px' }} />
                    <ClayIconButton
                        size="sm"
                        icon={<ChevronDown />}
                        onClick={() => setIsCollapsed(true)}
                        title="Collapse input area"
                    />
                </div>
            </div>
        </div>

        {isCollapsed && (
            <div className="composer-restore">
                <ClayIconButton
                    size="sm"
                    icon={<ChevronUp />}
                    onClick={() => setIsCollapsed(false)}
                    title="Restore input area"
                />
            </div>
        )}
        </>
    );
}
