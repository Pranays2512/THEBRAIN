import EventBus from "../../lib/runtime/EventBus";

import { SESSION_EVENTS } from "./SessionEvents";

import {

    createSessionState,

    SESSION_STATUS

} from "./SessionState";

export default class SessionEngine{

    constructor({

        provider

    }){

        this.provider = provider;

        this.state = createSessionState();

    }

    getState(){

        return this.state;

    }

    async send(

        messages,

        onToken

    ){

        this.state = {

            ...this.state,

            status: SESSION_STATUS.STREAMING,

            startedAt: Date.now()

        };

        EventBus.emit(

            SESSION_EVENTS.STATUS,

            this.state

        );

        await this.provider.stream(

            messages,

            onToken

        );

        this.state = {

            ...this.state,

            status: SESSION_STATUS.IDLE,

            finishedAt: Date.now()

        };

        EventBus.emit(

            SESSION_EVENTS.STATUS,

            this.state

        );

    }

    cancel(){

        this.provider.abort?.();

        this.state = {

            ...this.state,

            status: SESSION_STATUS.CANCELLED

        };

        EventBus.emit(

            SESSION_EVENTS.STATUS,

            this.state

        );

    }

}
