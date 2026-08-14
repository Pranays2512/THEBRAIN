import { useEffect, useState } from "react";

let globalState = {
    somActivation: Array(64).fill(0),
    emotion: { arousal: 0.5, valence: 0.8, novelty: 0.5, surprise: 0.0 },
    workingMemory: { subject: "--", relation: "--", object: "--" },
    answerMeta: { kind: "none", verified: false },
    arousalHistory: [0.3, 0.4, 0.5, 0.6, 0.5, 0.7, 0.6, 0.8, 0.7, 0.5],
    valenceHistory: [0.8, 0.8, 0.85, 0.9, 0.8, 0.85, 0.9, 0.95, 0.9, 0.8],
    noveltyHistory: [0.5, 0.6, 0.4, 0.7, 0.8, 0.5, 0.6, 0.7, 0.5, 0.6],
    surpriseHistory: [0.0, 0.1, 0.0, 0.2, 0.0, 0.1, 0.3, 0.0, 0.1, 0.0]
};

const listeners = new Set();

function notify() {
    listeners.forEach(fn => fn({ ...globalState }));
}

let ws = null;
function initWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return;
    }

    try {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const host = window.location.hostname === "localhost" ? "localhost:8000" : window.location.host;
        ws = new WebSocket(`${protocol}//${host}/ws`);

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === "state") {
                    if (data.som_activation) globalState.somActivation = data.som_activation;
                    if (data.emotion) {
                        globalState.emotion = { ...globalState.emotion, ...data.emotion };
                        globalState.arousalHistory = [...globalState.arousalHistory.slice(1), data.emotion.arousal || 0.5];
                        globalState.valenceHistory = [...globalState.valenceHistory.slice(1), data.emotion.valence || 0.8];
                        globalState.noveltyHistory = [...globalState.noveltyHistory.slice(1), data.emotion.novelty || 0.5];
                        globalState.surpriseHistory = [...globalState.surpriseHistory.slice(1), data.emotion.surprise || 0.0];
                    }
                    if (data.working_memory) globalState.workingMemory = data.working_memory;
                    if (data.answer_meta) globalState.answerMeta = data.answer_meta;
                    notify();
                }
            } catch (err) {
                console.error("Telemetry decode error", err);
            }
        };

        ws.onclose = () => {
            ws = null;
            setTimeout(initWebSocket, 3000);
        };
    } catch (e) {
        console.error("WS init error", e);
    }
}

export default function useTelemetry() {
    const [state, setState] = useState(globalState);

    useEffect(() => {
        initWebSocket();
        listeners.add(setState);
        return () => {
            listeners.delete(setState);
        };
    }, []);

    return state;
}
