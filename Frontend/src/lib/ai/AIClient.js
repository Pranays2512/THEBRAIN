export default class AIClient {

    constructor(provider){

        this.provider = provider;

    }

    async stream(messages,onToken){

        return this.provider.stream(

            messages,

            onToken

        );

    }

    async models(){

        return this.provider.models();

    }

    abort(){

        this.provider.abort?.();

    }

}
