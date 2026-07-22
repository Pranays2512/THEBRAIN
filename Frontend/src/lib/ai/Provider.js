export default class Provider{

    async stream(){

        throw new Error(

            "stream() not implemented."

        );

    }

    async models(){

        return [];

    }

    abort(){}

}
