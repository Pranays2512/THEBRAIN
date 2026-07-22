export default class Scheduler{

    constructor({

        registry,

        queue

    }){

        this.registry=registry;

        this.queue=queue;

    }

    schedule(){

        const worker=

            this.registry.available()[0];

        if(!worker)

            return;

        const job=

            this.queue.pop();

        if(!job)

            return;

        worker.run(job);

    }

}
