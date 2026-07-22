import { useEffect } from "react";

export default function useAutoScroll(

    virtualizer,

    messages

){

    useEffect(()=>{

        virtualizer.scrollToIndex(

            messages.length-1,

            {

                align:"end"

            }

        );

    },[messages,virtualizer]);

}
