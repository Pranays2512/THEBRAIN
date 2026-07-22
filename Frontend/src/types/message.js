export function createMessage({

    role,

    blocks = [],

    status = "complete",

    metadata = {}

}){

    return{

        id:crypto.randomUUID(),

        role,

        blocks,

        status,

        metadata,

        createdAt:Date.now(),

        updatedAt:Date.now()

    };

}
