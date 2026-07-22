class EventBus{

    constructor(){

        this.listeners=new Map();

    }

    subscribe(type,callback){

        if(!this.listeners.has(type)){

            this.listeners.set(type,new Set());

        }

        this.listeners.get(type).add(callback);

        return()=>{

            this.listeners.get(type)?.delete(callback);

        };

    }

    emit(type,payload){

        this.listeners

            .get(type)

            ?.forEach(listener=>listener(payload));

    }

}

export default new EventBus();
