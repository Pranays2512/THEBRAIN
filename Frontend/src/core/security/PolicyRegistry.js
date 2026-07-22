export default class PolicyRegistry{

    constructor(){

        this.rules=[];

    }

    register(rule){

        this.rules.push(rule);

    }

    evaluate(context){

        for(const rule of this.rules){

            const result=

                rule(context);

            if(result)

                return result;

        }

        return{

            allow:false,

            reason:"Denied"

        };

    }

}
