export default class JobQueue{

    constructor(){

        this.queue=[];

    }

    push(job){

        this.queue.push(job);

    }

    pop(){

        return this.queue.shift();

    }

    size(){

        return this.queue.length;

    }

}
