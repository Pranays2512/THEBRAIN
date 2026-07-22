export default class Memory{

    constructor(){

        this.entries=[];

    }

    remember(item){

        this.entries.push(item);

    }

    recall(){

        return this.entries;

    }

}
