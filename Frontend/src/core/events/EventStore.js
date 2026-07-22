export default class EventStore{

    constructor(){

        this.events=[];

    }

    append(event){

        this.events.push({

            id:crypto.randomUUID(),

            timestamp:Date.now(),

            ...event

        });

    }

    all(){

        return this.events;

    }

    since(timestamp){

        return this.events.filter(

            e=>e.timestamp>=timestamp

        );

    }

}
