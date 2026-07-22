export default class WorkflowExecutor{

    constructor({

        tools

    }){

        this.tools=tools;

    }

    async execute(node){

        node.status="running";

        const tool = this.tools.get(node.tool);

        if (!tool) {
            throw new Error(`Tool ${node.tool} not found.`);
        }

        // Support both function-based tools and object-based tools (.invoke)
        if (typeof tool === "function") {
            node.output = await tool(node.input);
        } else if (typeof tool.invoke === "function") {
            node.output = await tool.invoke(node.input);
        } else {
            throw new Error(`Tool ${node.tool} is not executable.`);
        }

        node.status="finished";

        return node.output;

    }

}
