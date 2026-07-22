export function createToolCall({

    tool,

    input,

    status="running"

}){

    return{

        id:crypto.randomUUID(),

        tool,

        input,

        output:null,

        status,

        startedAt:Date.now(),

        finishedAt:null

    };

}
