export const TASK_STATUS = {

    QUEUED: "queued",

    RUNNING: "running",

    COMPLETED: "completed",

    FAILED: "failed",

    CANCELLED: "cancelled"

};

export function createTask({

    type,

    title,

    payload

}){

    return{

        id: crypto.randomUUID(),

        type,

        title,

        payload,

        progress:0,

        status:TASK_STATUS.QUEUED,

        createdAt:Date.now(),

        startedAt:null,

        finishedAt:null,

        result:null,

        error:null

    };

}
