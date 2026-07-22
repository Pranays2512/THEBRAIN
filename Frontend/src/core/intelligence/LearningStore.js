export default class LearningStore{

    constructor(){

        this.records=[];

    }

    add(record){

        this.records.push({

            id:crypto.randomUUID(),

            timestamp:Date.now(),

            ...record

        });

    }

    query(predicate){

        return this.records.filter(predicate);

    }

}
