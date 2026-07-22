import MCPRegistry from "./MCPRegistry";
import CommandRouter from "../../lib/commands/CommandRouter";

export default class MCPClient {

    constructor(){

        this.connections=[];

    }

    async add(connection){

        await connection.connect();

        this.connections.push(connection);

    }

    async discover(connection){

        if (typeof connection.transport.listTools === "function") {
            const tools = await connection.transport.listTools();
            tools.forEach(tool => {
                MCPRegistry.registerTool(tool);

                CommandRouter.register({
                    id: `mcp.${tool.name}`,
                    title: tool.name,
                    category: "MCP",
                    run(args) {
                        return tool.invoke(args);
                    }
                });
            });
        }

    }

}
