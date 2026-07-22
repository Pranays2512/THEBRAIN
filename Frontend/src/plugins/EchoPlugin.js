import manager from "./PluginManager";
import EventBus from "../lib/runtime/EventBus";
import { EVENTS } from "../lib/runtime/eventTypes";

const plugin = {
    id: "echo",
    name: "Echo",
    version: "1.0.0",
    activate(api) {
        api.registerTool("echo", async (input) => {
            EventBus.emit(EVENTS.TOOL_PROGRESS, { tool: "echo", message: "Searching..." });
            await new Promise(r => setTimeout(r, 1000));
            EventBus.emit(EVENTS.TOOL_PROGRESS, { tool: "echo", message: "Summarizing..." });
            await new Promise(r => setTimeout(r, 1000));
            return { text: `Echo: ${input.text}` };
        });

        api.registerCommand({
            id: "chat.clear",
            title: "Clear Chat",
            category: "Chat",
            aliases: ["reset", "empty"],
            run: () => {
                console.log("Chat cleared!");
            }
        });
        
        api.registerCommand({
            id: "settings.theme",
            title: "Toggle Theme",
            category: "Settings",
            aliases: ["dark mode", "light mode"],
            run: () => {
                console.log("Theme toggled!");
            }
        });
    }
};

manager.register(plugin);

export default plugin;
