import TextBlock from "./TextBlock";
import ImageBlock from "./ImageBlock";
import FileBlock from "./FileBlock";
import ToolBlock from "./ToolBlock";

export default function BlockRenderer({

blocks

}){

return(

<>

{

blocks.map(block=>{

switch(block.type){

case"text":

return(

<TextBlock

key={block.id}

block={block}

/>

);

case"image":

return(

<ImageBlock

key={block.id}

block={block}

/>

);

case"file":

return(

<FileBlock

key={block.id}

block={block}

/>

);

case"tool":

return(

<ToolBlock

key={block.id}

block={block}

/>

);

default:

return null;

}

})

}

</>

);

}
