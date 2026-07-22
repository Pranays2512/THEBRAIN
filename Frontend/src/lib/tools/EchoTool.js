import{

registerTool

}from"./ToolRegistry";
import EventBus from "../runtime/EventBus";
import { EVENTS } from "../runtime/eventTypes";

registerTool(

"echo",

async(input)=>{

EventBus.emit(EVENTS.TOOL_PROGRESS, { tool: "echo", message: "Searching..." });

await new Promise(

r=>setTimeout(r,1000)

);

EventBus.emit(EVENTS.TOOL_PROGRESS, { tool: "echo", message: "Summarizing..." });

await new Promise(

r=>setTimeout(r,1000)

);

const result = {

text:

`Echo: ${input.text}`

};

EventBus.emit(EVENTS.TOOL_END, result);

return result;

}

);
