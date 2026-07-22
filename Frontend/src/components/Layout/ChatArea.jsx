import WelcomeHero from "../Home/WelcomeHero";
import ChatView from "../Chat/ChatView";
import Composer from "../Composer/Composer";
import useChat from "../../hooks/useChat";
import "./ChatArea.css";

export default function ChatArea() {
    const { messages, sendMessage, streaming, stop } = useChat();

    return (
        <div className="chat-area-layout">
            <div className="chat-area-messages">
                {messages.length === 0 ? (
                    <WelcomeHero />
                ) : (
                    <ChatView messages={messages} />
                )}
            </div>
            <div className="chat-area-composer">
                <Composer
                    onSend={sendMessage}
                    streaming={streaming}
                    onStop={stop}
                />
            </div>
        </div>
    );
}
