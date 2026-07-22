import Provider

from "../lib/ai/Provider";

function delay(ms){

    return new Promise(resolve=>

        setTimeout(resolve,ms)

    );

}

export default class MockProvider

extends Provider{

    async stream(

        messages,

        onToken

    ){
        const lastMsg = messages.at(-1);
        const last = lastMsg?.blocks?.[0]?.text || lastMsg?.content || "";

        const answer=

`# Mock Response

You asked:

> ${last}

This is currently coming from the MockProvider.

Replacing this provider with OpenAI or Ollama won't require changing the UI.`;

        let current="";

        for(

            const ch of answer

        ){

            current+=ch;

            onToken(current);

            await delay(12);

        }

    }

    async models(){

        return [

            "mock"

        ];

    }

}
