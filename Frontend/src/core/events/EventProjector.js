export default class EventProjector{

    constructor(initial={}){

        this.state=initial;

        this.handlers=new Map();

    }

    on(type,handler){

        this.handlers.set(type,handler);

    }

    project(event){

        const handler=

            this.handlers.get(event.type);

        if(handler){

            this.state=

                handler(

                    this.state,

                    event

                );

        }

    }

    getState(){

        return this.state;

    }

}
