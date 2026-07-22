export default async function streamText(

    text,

    callback,

    speed=12

){

    let current="";

    for(

        let i=0;

        i<text.length;

        i++

    ){

        current+=text[i];

        callback(current);

        await new Promise(r=>

            setTimeout(

                r,

                speed

            )

        );

    }

}
