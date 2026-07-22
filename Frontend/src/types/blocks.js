export function TextBlock(text){

    return{

        id:crypto.randomUUID(),

        type:"text",

        text

    };

}

export function ImageBlock({

    url,

    alt=""

}){

    return{

        id:crypto.randomUUID(),

        type:"image",

        url,

        alt

    };

}

export function FileBlock({

    name,

    url,

    mime,

    size

}){

    return{

        id:crypto.randomUUID(),

        type:"file",

        name,

        url,

        mime,

        size

    };

}

export function ToolBlock(toolCall){

    return{

        id:crypto.randomUUID(),

        type:"tool",

        toolCall

    };

}
