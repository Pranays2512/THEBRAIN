export default class Metrics{

    constructor(){

        this.counters=new Map();

    }

    increment(name,value=1){

        this.counters.set(

            name,

            (this.counters.get(name)||0)+value

        );

    }

    value(name){

        return this.counters.get(name)||0;

    }

}
