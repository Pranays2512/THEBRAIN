import Span from "./Span";

export default class Tracer{

    constructor(){

        this.spans=[];

    }

    start(name){

        const span=

            new Span(name);

        this.spans.push(span);

        return span;

    }

}
