export default class WorkerRegistry{

    constructor(){

        this.workers=new Map();

    }

    register(worker){

        this.workers.set(

            worker.id,

            worker

        );

    }

    unregister(id){

        this.workers.delete(id);

    }

    available(){

        return [...this.workers.values()]

            .filter(

                w=>w.status==="idle"

            );

    }

}
