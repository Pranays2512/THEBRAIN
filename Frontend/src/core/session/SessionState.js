export const SESSION_STATUS = {

    IDLE: "idle",

    STREAMING: "streaming",

    TOOL: "tool",

    CANCELLED: "cancelled",

    ERROR: "error"

};

export function createSessionState(){

    return{

        id: crypto.randomUUID(),

        status: SESSION_STATUS.IDLE,

        provider: "fastapi",

        model: null,

        startedAt: null,

        finishedAt: null,

        tokens:{

            prompt:0,

            completion:0,

            total:0

        }

    };

}
