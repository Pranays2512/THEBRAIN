import EventBus from "./EventBus";
import { EVENTS } from "./eventTypes";

export function streamText(text){

    EventBus.emit(EVENTS.MESSAGE_START);

    let current="";

    let i=0;

    const timer=setInterval(()=>{

        current+=text[i];

        EventBus.emit(

            EVENTS.MESSAGE_TOKEN,

            current

        );

        i++;

        if(i>=text.length){

            clearInterval(timer);

            EventBus.emit(

                EVENTS.MESSAGE_END,

                current

            );

        }

    },12);

}
