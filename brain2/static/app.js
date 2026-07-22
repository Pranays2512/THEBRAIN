document.addEventListener("DOMContentLoaded", () => {
    const chatHistory = document.getElementById("chat-history");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const statusIndicator = document.getElementById("status-indicator");

    // --- WEBSOCKET ENGINE (For Neural Status) ---
    let ws;
    function connect() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
        
        ws.onopen = () => {
            // connected
            statusIndicator.classList.remove("thinking");
        };
        
        ws.onclose = () => {
            addLog("System connection severed. Retrying...", "brain-msg");
            setTimeout(connect, 2000);
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "state" && data.som_activation) {
                // Determine total activation spike
                const spike = data.som_activation.reduce((a, b) => a + b, 0);
                
                // If there's a significant spike, make the status indicator glow cyan
                if (spike > 0.5) {
                    statusIndicator.classList.add("thinking");
                    
                    // Settle down after a moment
                    setTimeout(() => {
                        statusIndicator.classList.remove("thinking");
                    }, 1000);
                }
            }
        };
    }
    connect();

    // --- CHAT LOGIC ---
    function addLog(text, typeClass) {
        // Wrapper for alignment
        const wrapper = document.createElement("div");
        wrapper.className = `msg-wrapper ${typeClass}-wrapper`;
        
        // The actual bubble
        const div = document.createElement("div");
        div.className = `msg ${typeClass}`;
        
        // Optional Markdown-like simple formatting (just plain text for now)
        div.textContent = text;
        
        wrapper.appendChild(div);
        chatHistory.appendChild(wrapper);
        
        // Smooth scroll to bottom
        chatHistory.scrollTo({
            top: chatHistory.scrollHeight,
            behavior: 'smooth'
        });
    }

    async function submitQuery() {
        const text = userInput.value.trim();
        if (!text) return;
        
        // Add user message
        addLog(text, "user-msg");
        userInput.value = "";
        
        // Instant visual feedback (thinking)
        statusIndicator.classList.add("thinking");

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text })
            });
            
            statusIndicator.classList.remove("thinking");

            const data = await res.json();
            if (data.reply) {
                addLog(data.reply, "brain-msg");
            }
        } catch (err) {
            statusIndicator.classList.remove("thinking");
            addLog("Critical Fault: API Unreachable.", "brain-msg");
        }
    }

    // Event Listeners
    userInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") submitQuery();
    });
    
    sendBtn.addEventListener("click", submitQuery);
});
