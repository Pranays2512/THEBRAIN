import PolicyRegistry
from "./PolicyRegistry";

export default class PolicyEngine{

    constructor(){

        this.registry=

            new PolicyRegistry();

    }

    authorize(context){

        return this.registry.evaluate(

            context

        );

    }

}
