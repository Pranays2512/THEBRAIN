import MarkdownRenderer from "../MarkdownRenderer";

export default function TextBlock({

block

}){

return(

<MarkdownRenderer>

{block.text}

</MarkdownRenderer>

);

}
