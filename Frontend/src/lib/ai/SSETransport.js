export default class SSETransport {

    constructor(baseURL){

        this.baseURL = baseURL;

        this.controller = null;

    }

    async stream(

        endpoint,

        body,

        onToken

    ){

        this.controller = new AbortController();

        const response = await fetch(

            `${this.baseURL}${endpoint}`,

            {

                method:"POST",

                headers:{

                    "Content-Type":"application/json"

                },

                body:JSON.stringify(body),

                signal:this.controller.signal

            }

        );

        if(!response.ok){

            throw new Error(

                "Streaming request failed."

            );

        }

        const reader =

            response.body.getReader();

        const decoder =

            new TextDecoder();

        let buffer = "";

        while(true){

            const {

                done,

                value

            } = await reader.read();

            if(done)

                break;

            buffer += decoder.decode(

                value,

                {

                    stream:true

                }

            );

            const lines =

                buffer.split("\n");

            buffer =

                lines.pop() ?? "";

            for(const line of lines){

                if(

                    !line.startsWith("data:")

                )

                    continue;

                const data =

                    line.replace(

                        "data:",

                        ""

                    ).trim();

                if(data==="[DONE]")

                    return;

                onToken(data);

            }

        }

    }

    abort(){

        this.controller?.abort();

    }

}
