import ChatMessage from "./ChatMessage";
import useVirtualChat from "../../hooks/useVirtualChat";
import useAutoScroll from "../../hooks/useAutoScroll";

import "./ChatView.css";

function getText(blocks = []) {
    return blocks
        .filter(block => block.type === "text")
        .map(block => block.text)
        .join("\n")
        .trim();
}

function findPromptForMessage(messages, index) {
    const message = messages[index];
    if (message.role === "user") return getText(message.blocks);

    for (let i = index - 1; i >= 0; i -= 1) {
        if (messages[i].role === "user") return getText(messages[i].blocks);
    }

    return "";
}

export default function ChatView({ messages, onResendMessage }) {

    const {

        parentRef,

        virtualizer

    } = useVirtualChat(messages);

    useAutoScroll(virtualizer, messages);

    return (

        <section

            className="chat-view"

            ref={parentRef}

        >

            <div
                style={{

                    height:
                        virtualizer.getTotalSize(),

                    position:"relative"

                }}
            >

                {

                    virtualizer
                        .getVirtualItems()
                        .map(item=>{

                            const message =
                                messages[item.index];

                            return(

                                <div

                                    key={message.id}

                                    ref={virtualizer.measureElement}

                                    data-index={item.index}

                                    style={{

                                        position:"absolute",

                                        top: 0,
                                        left: 0,
                                        width:"100%",

                                        transform:
                                            `translateY(${item.start}px)`

                                    }}

                                >

                                    <ChatMessage

                                        {...message}

                                        promptText={findPromptForMessage(messages, item.index)}

                                        onResendMessage={onResendMessage}

                                    />

                                </div>

                            );

                        })

                }

            </div>

        </section>

    );

}
