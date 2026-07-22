import PluginManager from "../../plugins/PluginManager";

export default class Executor{

    async execute(step){

        const tool = PluginManager.get("tools").get(step.tool);
        if (!tool) {
            throw new Error(`Tool ${step.tool} not found.`);
        }
        
        return tool(step.arguments || {});
    }

}
