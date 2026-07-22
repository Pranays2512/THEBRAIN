import ChatMessage from "./ChatMessage";
import useVirtualChat from "../../hooks/useVirtualChat";
import useAutoScroll from "../../hooks/useAutoScroll";

import "./ChatView.css";

export default function ChatView({ messages }) {

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

                                    />

                                </div>

                            );

                        })

                }

            </div>

        </section>

    );

}
