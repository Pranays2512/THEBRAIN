export default class WorkflowGraph{

    constructor(){

        this.nodes=new Map();

    }

    add(node){

        this.nodes.set(

            node.id,

            node

        );

    }

    get(id){

        return this.nodes.get(id);

    }

    list(){

        return [...this.nodes.values()];

    }

}
