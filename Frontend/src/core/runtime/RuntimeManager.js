export default class RuntimeManager{

    constructor({

        scheduler,

        registry,

        queue

    }){

        this.scheduler=scheduler;

        this.registry=registry;

        this.queue=queue;

    }

    submit(job){

        this.queue.push(job);

        this.scheduler.schedule();

    }

}
