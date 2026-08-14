import Provider from "../lib/ai/Provider";
import SSETransport from "../lib/ai/SSETransport";

export default class FastAPIProvider extends Provider{

    constructor(){
        super();
        this.transport = new SSETransport("http://localhost:8000");
    }

    async stream(messages, onToken) {
        let fullText = "";
        await this.transport.stream(
            "/chat/stream",
            { messages },
            (data) => {
                try {
                    const parsed = JSON.parse(data);
                    if (parsed && typeof parsed === "object") {
                        if (parsed.text) {
                            fullText += parsed.text;
                            onToken(fullText);
                        }
                    } else if (typeof parsed === "string") {
                        fullText += parsed;
                        onToken(fullText);
                    }
                } catch(e) {
                    if (data && data !== "[DONE]") {
                        fullText += data;
                        onToken(fullText);
                    }
                }
            }
        );
    }

    abort(){
        this.transport.abort();
    }
}

