export function createArtifact({

    type,

    title,

    data,

    metadata={}

}){

    return{

        id:crypto.randomUUID(),

        type,

        title,

        data,

        metadata,

        createdAt:Date.now()

    };

}
