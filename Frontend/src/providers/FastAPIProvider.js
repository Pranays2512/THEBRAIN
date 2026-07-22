import Provider from "../lib/ai/Provider";
import SSETransport from "../lib/ai/SSETransport";

export default class FastAPIProvider extends Provider{

    constructor(){

        super();

        this.transport =

            new SSETransport(

                "http://localhost:8000"

            );

    }

    async stream(

        messages,

        onToken

    ){

        await this.transport.stream(

            "/chat/stream",

            {

                messages

            },

            onToken

        );

    }

    abort(){

        this.transport.abort();

    }

}
