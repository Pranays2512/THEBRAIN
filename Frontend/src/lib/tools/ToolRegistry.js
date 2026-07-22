import EventBus from "../runtime/EventBus";
import { EVENTS } from "../runtime/eventTypes";
import PluginManager from "../../plugins/PluginManager";

export async function executeTool(name, input) {
    const handler = PluginManager.get("tools").get(name);
    
    if (!handler) {
        throw new Error(`Unknown tool: ${name}`);
    }

    EventBus.emit(EVENTS.TOOL_START, { tool: name, input });

    try {
        const result = await handler(input);
        EventBus.emit(EVENTS.TOOL_END, result);
        return result;
    } catch (error) {
        EventBus.emit(EVENTS.ERROR, error);
        throw error;
    }
}
