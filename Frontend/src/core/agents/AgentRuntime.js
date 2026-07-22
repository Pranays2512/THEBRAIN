import EventBus from "../../lib/runtime/EventBus";

import {
AGENT_STATUS,
createAgentState
} from "./AgentState";
import { AGENT_EVENTS } from "./AgentEvents";
import WorkflowEngine from "../workflow/WorkflowEngine";
import WorkflowExecutor from "../workflow/WorkflowExecutor";
import PluginManager from "../../plugins/PluginManager";

export default class AgentRuntime{

    constructor({
        planner,
        memory
    }){
        this.planner=planner;
        this.memory=memory;
        this.state=createAgentState();
        this.workflowEngine = new WorkflowEngine({
            executor: new WorkflowExecutor({
                tools: PluginManager.get("tools")
            })
        });
    }

    async run(input){

        this.state.status=

            AGENT_STATUS.PLANNING;

        EventBus.emit(

            AGENT_EVENTS.STATUS_CHANGED,

            this.state

        );

        this.state.plan=

            await this.planner.createPlan(input);

        EventBus.emit(

            AGENT_EVENTS.PLAN_CREATED,

            this.state.plan

        );

        this.state.status=
            AGENT_STATUS.EXECUTING;

        // Plan is now a WorkflowGraph
        await this.workflowEngine.run(this.state.plan);

        this.state.status=
            AGENT_STATUS.FINISHED;
            
        EventBus.emit(
            AGENT_EVENTS.STATUS_CHANGED,
            this.state
        );

    }

}
