import { useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";

export default function useVirtualChat(messages){

    const parentRef = useRef(null);

    const virtualizer = useVirtualizer({

        count: messages.length,

        getScrollElement: () => parentRef.current,

        estimateSize: () => 150,

        overscan: 8

    });

    return{

        parentRef,

        virtualizer

    };

}
