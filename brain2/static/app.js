document.addEventListener("DOMContentLoaded", () => {
    const chatHistory = document.getElementById("chat-history");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const statusIndicator = document.getElementById("status-indicator");
    
    // Telemetry Elements
    const somGrid = document.getElementById("som-grid");
    const valArousal = document.getElementById("val-arousal");
    const barArousal = document.getElementById("bar-arousal");
    const valValence = document.getElementById("val-valence");
    const barValence = document.getElementById("bar-valence");
    const wmSubject = document.getElementById("wm-subject");
    const wmRelation = document.getElementById("wm-relation");
    const wmObject = document.getElementById("wm-object");
    const teachForm = document.getElementById("teach-form");
    const teachStatus = document.getElementById("teach-status");

    // Initialize 8x8 SOM Grid
    const somCells = [];
    if (somGrid) {
        somGrid.innerHTML = "";
        for (let i = 0; i < 64; i++) {
            const cell = document.createElement("div");
            cell.className = "som-cell";
            cell.title = `Neuron ${i}`;
            somGrid.appendChild(cell);
            somCells.push(cell);
        }
    }

    // --- WEBSOCKET ENGINE (For Neural Telemetry) ---
    let ws;
    function connect() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
        
        ws.onopen = () => {
            if (statusIndicator) statusIndicator.classList.remove("thinking");
        };
        
        ws.onclose = () => {
            setTimeout(connect, 2000);
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === "state") {
                    updateTelemetry(data);
                }
            } catch (err) {
                console.error("WS decode error", err);
            }
        };
    }
    connect();

    function updateTelemetry(data) {
        // 1. SOM Grid Activation
        if (data.som_activation && data.som_activation.length === 64) {
            for (let i = 0; i < 64; i++) {
                const val = data.som_activation[i];
                const cell = somCells[i];
                if (cell) {
                    if (val > 0.05) {
                        cell.style.backgroundColor = `rgba(0, 229, 196, ${Math.min(val * 0.9 + 0.1, 1.0)})`;
                        cell.classList.add("active");
                    } else {
                        cell.style.backgroundColor = "rgba(160, 165, 176, 0.2)";
                        cell.classList.remove("active");
                    }
                }
            }
        }

        // 2. Emotion & Cognitive Gauges
        if (data.emotion) {
            const arousal = Math.max(0, Math.min(1, data.emotion.arousal || 0));
            const valence = Math.max(0, Math.min(1, Math.abs(data.emotion.valence || 0)));
            
            if (valArousal) valArousal.textContent = arousal.toFixed(2);
            if (barArousal) barArousal.style.width = `${Math.round(arousal * 100)}%`;
            
            if (valValence) valValence.textContent = (data.emotion.valence || 0).toFixed(2);
            if (barValence) barValence.style.width = `${Math.round(valence * 100)}%`;
        }

        // 3. Working Memory
        if (data.working_memory) {
            if (wmSubject) wmSubject.textContent = data.working_memory.subject || "--";
            if (wmRelation) wmRelation.textContent = data.working_memory.relation || "--";
            if (wmObject) wmObject.textContent = data.working_memory.object || "--";
        }

        // Quick indicator glow
        if (statusIndicator) {
            statusIndicator.classList.add("thinking");
            setTimeout(() => statusIndicator.classList.remove("thinking"), 800);
        }
    }

    // --- CHAT LOGIC WITH SSE STREAMING ---
    function createMessageBubble(typeClass) {
        const wrapper = document.createElement("div");
        wrapper.className = `msg-wrapper ${typeClass}-wrapper`;
        
        const div = document.createElement("div");
        div.className = `msg ${typeClass}`;
        
        wrapper.appendChild(div);
        chatHistory.appendChild(wrapper);
        return { wrapper, div };
    }

    function scrollToBottom() {
        chatHistory.scrollTo({
            top: chatHistory.scrollHeight,
            behavior: 'smooth'
        });
    }

    async function submitQuery() {
        const text = userInput.value.trim();
        if (!text) return;
        
        // 1. Render User Message
        const userBubble = createMessageBubble("user-msg");
        userBubble.div.textContent = text;
        userInput.value = "";
        scrollToBottom();
        
        // Visual indicator
        if (statusIndicator) statusIndicator.classList.add("thinking");

        // 2. Create Brain Streaming Message Bubble
        const brainBubble = createMessageBubble("brain-msg");
        const textNode = document.createTextNode("");
        brainBubble.div.appendChild(textNode);
        
        // Metadata footer element
        const metaDiv = document.createElement("div");
        metaDiv.className = "msg-meta";
        brainBubble.div.appendChild(metaDiv);

        try {
            const response = await fetch("/chat/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    messages: [{ role: "user", content: text }]
                })
            });

            if (!response.ok) {
                textNode.nodeValue = "Error connecting to Brain server.";
                if (statusIndicator) statusIndicator.classList.remove("thinking");
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed.startsWith("data: ")) continue;
                    const dataStr = trimmed.slice(6);
                    
                    if (dataStr === "[DONE]") break;
                    
                    try {
                        const parsed = JSON.parse(dataStr);
                        
                        // Handle metadata header
                        if (parsed.meta) {
                            metaDiv.innerHTML = "";
                            
                            const kindBadge = document.createElement("span");
                            kindBadge.className = "badge badge-kind";
                            kindBadge.textContent = parsed.meta.kind || "language";
                            metaDiv.appendChild(kindBadge);

                            const verifyBadge = document.createElement("span");
                            verifyBadge.className = `badge ${parsed.meta.verified ? 'badge-verified' : 'badge-unverified'}`;
                            verifyBadge.textContent = parsed.meta.verified ? "VERIFIED ✓" : "UNVERIFIED";
                            metaDiv.appendChild(verifyBadge);
                        }
                        
                        // Stream text chunks
                        if (parsed.text) {
                            textNode.nodeValue += parsed.text;
                            scrollToBottom();
                        }
                    } catch (err) {
                        console.error("SSE parse error", err);
                    }
                }
            }

        } catch (err) {
            console.error(err);
            textNode.nodeValue = "Critical Fault: Brain API unreachable.";
        } finally {
            if (statusIndicator) statusIndicator.classList.remove("thinking");
            scrollToBottom();
        }
    }

    // --- TEACH FORM HANDLER ---
    if (teachForm) {
        teachForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const subj = document.getElementById("teach-subj").value.trim();
            const rel = document.getElementById("teach-rel").value.trim();
            const obj = document.getElementById("teach-obj").value.trim();
            
            if (!subj || !rel || !obj) return;
            
            if (teachStatus) teachStatus.textContent = "Teaching brain...";
            
            try {
                const res = await fetch("/api/teach", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ subject: subj, relation: rel, object: obj })
                });
                const data = await res.json();
                if (teachStatus) {
                    teachStatus.textContent = data.message || "Learned!";
                    setTimeout(() => { teachStatus.textContent = ""; }, 3000);
                }
                document.getElementById("teach-subj").value = "";
                document.getElementById("teach-rel").value = "";
                document.getElementById("teach-obj").value = "";
            } catch (err) {
                if (teachStatus) teachStatus.textContent = "Failed to teach.";
            }
        });
    }

    // Event Listeners
    if (userInput) {
        userInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") submitQuery();
        });
    }
    
    if (sendBtn) {
        sendBtn.addEventListener("click", submitQuery);
    }
});

