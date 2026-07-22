import {
    createContext,
    useContext,
    useEffect,
    useMemo,
    useState
} from "react";

const ChatContext = createContext();

const STORAGE_KEY = "brain-ai-chats";

function createConversation(title = "New Chat") {
    return {
        id: crypto.randomUUID(),
        title,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [
            {
                id: crypto.randomUUID(),
                role: "assistant",
                blocks: [
                    {
                        id: crypto.randomUUID(),
                        type: "text",
                        text: "# Welcome\n\nHow can I help today?"
                    }
                ],
                status: "complete",
                createdAt: Date.now()
            }
        ]
    };
}

export function ChatProvider({ children }) {

    const [conversations, setConversations] = useState(() => {

        const saved = localStorage.getItem(STORAGE_KEY);

        if (saved) {

            try {

                return JSON.parse(saved);

            } catch {}

        }

        return [createConversation()];

    });

    const [activeId, setActiveId] = useState(
        conversations[0].id
    );

    useEffect(() => {

        localStorage.setItem(

            STORAGE_KEY,

            JSON.stringify(conversations)

        );

    }, [conversations]);

    const activeConversation = useMemo(

        () =>
            conversations.find(
                c => c.id === activeId
            ),

        [conversations, activeId]

    );

    function newConversation() {

        const chat = createConversation();

        setConversations(prev => [

            chat,

            ...prev

        ]);

        setActiveId(chat.id);

    }

    function deleteConversation(id) {

        setConversations(prev => {

            const next = prev.filter(c => c.id !== id);

            if (next.length === 0)
                return [createConversation()];

            if (id === activeId)
                setActiveId(next[0].id);

            return next;

        });

    }

    function renameConversation(id, title) {

        setConversations(prev =>

            prev.map(chat =>

                chat.id === id

                    ? {

                        ...chat,

                        title,

                        updatedAt: Date.now()

                    }

                    : chat

            )

        );

    }

    return (

        <ChatContext.Provider

            value={{

                conversations,

                setConversations,

                activeConversation,

                activeId,

                setActiveId,

                newConversation,

                deleteConversation,

                renameConversation

            }}

        >

            {children}

        </ChatContext.Provider>

    );

}

export function useChatContext(){

    return useContext(ChatContext);

}
