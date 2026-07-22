export default class WorkflowNode{

    constructor({

        id,

        type,

        tool,

        input={},

        dependsOn=[]

    }){

        this.id=id;

        this.type=type;

        this.tool=tool;

        this.input=input;

        this.dependsOn=dependsOn;

        this.output=null;

        this.status="waiting";

    }

}
