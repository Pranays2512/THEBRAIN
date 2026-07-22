export default class Span{

    constructor(name){

        this.name=name;

        this.startedAt=performance.now();

        this.finishedAt=null;

    }

    finish(){

        this.finishedAt=

            performance.now();

    }

    duration(){

        return this.finishedAt-

            this.startedAt;

    }

}
