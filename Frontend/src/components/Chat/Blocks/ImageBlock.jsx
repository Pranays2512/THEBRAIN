export default function ImageBlock({

block

}){

return(

<img

className="chat-image"

src={block.url}

alt={block.alt}

/>

);

}
