import EventBus
from "../../lib/runtime/EventBus";

import{

WORKFLOW_EVENTS

}

from "./WorkflowEvents";

export default class WorkflowEngine{

constructor({

executor

}){

this.executor=executor;

}

async run(graph){

EventBus.emit(

WORKFLOW_EVENTS.START,

graph

);

const remaining=

new Set(

graph.list()

);

while(

remaining.size

){

const ready=

[...remaining]

.filter(node=>

node.dependsOn.every(

id=>

graph

.get(id)

.status==="finished"

)

);

if (ready.length === 0) {
    throw new Error("Deadlock detected: unresolved dependencies.");
}

await Promise.all(

ready.map(

async node=>{

remaining.delete(node);

EventBus.emit(

WORKFLOW_EVENTS.NODE_STARTED,

node

);

await this.executor.execute(node);

EventBus.emit(

WORKFLOW_EVENTS.NODE_FINISHED,

node

);

}

)

);

}

EventBus.emit(

WORKFLOW_EVENTS.COMPLETED,

graph

);

}

}
