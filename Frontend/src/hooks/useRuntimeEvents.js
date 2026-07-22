import { useEffect } from "react";

import EventBus from "../lib/runtime/EventBus";
import { EVENTS } from "../lib/runtime/eventTypes";

export default function useRuntimeEvents({

    onMessageToken,

    onToolProgress,

    onError

}){

    useEffect(()=>{

        const unsubscribers=[

            EventBus.subscribe(

                EVENTS.MESSAGE_TOKEN,

                onMessageToken

            ),

            EventBus.subscribe(

                EVENTS.TOOL_PROGRESS,

                onToolProgress

            ),

            EventBus.subscribe(

                EVENTS.ERROR,

                onError

            )

        ];

        return()=>{

            unsubscribers.forEach(unsub=>unsub());

        };

    },[

        onMessageToken,

        onToolProgress,

        onError

    ]);

}
