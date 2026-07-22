export const AGENT_STATUS = {

    IDLE:"idle",

    PLANNING:"planning",

    EXECUTING:"executing",

    REFLECTING:"reflecting",

    FINISHED:"finished",

    ERROR:"error"

};

export function createAgentState(){

    return{

        id:crypto.randomUUID(),

        status:AGENT_STATUS.IDLE,

        plan:[],

        currentStep:0,

        observations:[],

        startedAt:null,

        finishedAt:null

    };

}
