document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatHistory = document.getElementById("chat-history");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const statusIndicator = document.getElementById("connection-status");
    const statusText = document.getElementById("status-text");
    
    const somGrid = document.getElementById("som-grid");
    const slotSubject = document.getElementById("slot-subject");
    const slotRelation = document.getElementById("slot-relation");
    const slotObject = document.getElementById("slot-object");
    
    const barArousal = document.getElementById("bar-arousal");
    const barValence = document.getElementById("bar-valence");

    // Initialize 8x8 SOM Grid
    const neurons = [];
    for (let i = 0; i < 64; i++) {
        const div = document.createElement("div");
        div.className = "neuron";
        somGrid.appendChild(div);
        neurons.push(div);
    }

    // Connect WebSocket
    let ws;
    function connect() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
        
        ws.onopen = () => {
            statusIndicator.className = "indicator connected";
            statusText.textContent = "Neural Link Active";
        };
        
        ws.onclose = () => {
            statusIndicator.className = "indicator disconnected";
            statusText.textContent = "Disconnected - Reconnecting...";
            setTimeout(connect, 2000);
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "state") {
                updateVisuals(data);
            }
        };
    }
    connect();

    function updateVisuals(data) {
        // Update Working Memory
        slotSubject.textContent = data.working_memory.subject || "--";
        slotRelation.textContent = data.working_memory.relation || "--";
        slotObject.textContent = data.working_memory.object || "--";

        // Update SOM (64 values)
        if (data.som_activation && data.som_activation.length === 64) {
            for (let i = 0; i < 64; i++) {
                const act = data.som_activation[i];
                // Map activation 0-1 to an opacity of the accent color
                if (act > 0.05) {
                    neurons[i].style.backgroundColor = `rgba(0, 242, 254, ${act})`;
                    neurons[i].style.boxShadow = `0 0 ${act * 10}px rgba(0, 242, 254, ${act})`;
                } else {
                    neurons[i].style.backgroundColor = "rgba(255,255,255,0.05)";
                    neurons[i].style.boxShadow = "none";
                }
            }
        }

        // Update Emotions
        if (data.emotion) {
            // Arousal: 0 to 1 -> 0% to 100%
            const arousalPct = Math.min(Math.max(data.emotion.arousal * 100, 0), 100);
            barArousal.style.width = `${arousalPct}%`;

            // Valence: -1 to +1 -> 0% to 100%
            const valencePct = Math.min(Math.max((data.emotion.valence + 1) * 50, 0), 100);
            barValence.style.width = `${valencePct}%`;
        }
    }

    // Chat functionality
    function addMessage(text, sender) {
        const div = document.createElement("div");
        div.className = `message ${sender}`;
        div.textContent = text;
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;
        
        addMessage(text, "user");
        userInput.value = "";
        
        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            if (data.reply) {
                addMessage(data.reply, "brain");
            }
        } catch (err) {
            addMessage("Error communicating with Brain API.", "system");
        }
    }

    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });
});
