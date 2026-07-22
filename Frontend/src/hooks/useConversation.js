import { useCallback } from "react";
import { useChatContext } from "../context/ChatContext";

export default function useConversation() {

    const {

        activeConversation,

        conversations,

        activeId,

        setConversations

    } = useChatContext();

    const updateConversation = useCallback(

        (updater)=>{

            setConversations(prev=>{

                return prev.map(chat=>{

                    if(chat.id!==activeId)
                        return chat;

                    return updater(chat);

                });

            });

        },

        [activeId,setConversations]

    );

    const addMessage = useCallback(

        (message)=>{

            updateConversation(chat=>({

                ...chat,

                updatedAt:Date.now(),

                messages:[

                    ...chat.messages,

                    message

                ]

            }));

        },

        [updateConversation]

    );

    const replaceMessage = useCallback(

        (id,newMessage)=>{

            updateConversation(chat=>({

                ...chat,

                messages:

                    chat.messages.map(msg=>

                        msg.id===id

                        ? newMessage
                        : msg

                    )

            }));

        },

        [updateConversation]

    );

    return{

        conversation:activeConversation,

        messages:

            activeConversation?.messages ?? [],

        addMessage,

        replaceMessage,

        conversations

    };

}
