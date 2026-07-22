import { useState, useRef } from "react";
import useConversation from "./useConversation";
import { useSession } from "../context/SessionContext";
import { createMessage } from "../types/message";
import { TextBlock, ImageBlock, FileBlock } from "../types/blocks";
import useRuntimeEvents from "./useRuntimeEvents";
import EventBus from "../lib/runtime/EventBus";
import { EVENTS } from "../lib/runtime/eventTypes";

export default function useChat() {
    const { messages, addMessage, replaceMessage } = useConversation();
    const [streaming, setStreaming] = useState(false);
    
    const activeAssistantMsg = useRef(null);

    const session = useSession();

    useRuntimeEvents({
        onMessageToken: (content) => {
            if (!activeAssistantMsg.current) return;
            replaceMessage(
                activeAssistantMsg.current.id,
                {
                    ...activeAssistantMsg.current,
                    blocks: [
                        TextBlock(content)
                    ],
                    status: "streaming"
                }
            );
        },
        onToolProgress: (payload) => {
            console.log("Tool Progress:", payload);
        },
        onError: console.error
    });

    async function sendMessage(payload) {
        const isObj = typeof payload === "object" && payload !== null;
        const text = isObj ? payload.text : payload;
        const attachments = isObj ? payload.attachments || [] : [];
        
        const blocks = [];
        if (text) blocks.push(TextBlock(text));
        
        for (const a of attachments) {
            if (a.type.startsWith("image")) {
                blocks.push(ImageBlock({ url: a.preview, alt: a.name }));
            } else {
                blocks.push(FileBlock({ name: a.name, url: "", mime: a.type, size: a.size }));
            }
        }

        const user = createMessage({
            role: "user",
            blocks,
            status: "complete"
        });
        addMessage(user);

        const assistant = createMessage({
            role: "assistant",
            blocks: [TextBlock("")],
            status: "streaming"
        });
        addMessage(assistant);

        activeAssistantMsg.current = assistant;
        setStreaming(true);

        let finalResponse = "";

        try {
            EventBus.emit(EVENTS.MESSAGE_START);
            
            await session.send(
                [...messages, user],
                current => {
                    finalResponse = current;
                    EventBus.emit(EVENTS.MESSAGE_TOKEN, current);
                }
            );

            EventBus.emit(EVENTS.MESSAGE_END, finalResponse);

            replaceMessage(assistant.id, {
                ...assistant,
                blocks: [TextBlock(finalResponse)],
                status: "complete",
                updatedAt: Date.now()
            });
        } catch (err) {
            const isCancelled = session.getState().status === "cancelled";
            EventBus.emit(EVENTS.ERROR, err);
            replaceMessage(assistant.id, {
                ...assistant,
                blocks: [TextBlock(finalResponse)],
                status: isCancelled ? "cancelled" : "error",
                updatedAt: Date.now()
            });
        } finally {
            setStreaming(false);
            activeAssistantMsg.current = null;
        }
    }

    return {
        messages,
        sendMessage,
        streaming,
        stop() {
            session.cancel();
            setStreaming(false);
        }
    };
}
