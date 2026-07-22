import MarkdownRenderer

from "../Chat/MarkdownRenderer";

export default function MarkdownArtifact({

artifact

}){

return(

<MarkdownRenderer>

{artifact.data.markdown}

</MarkdownRenderer>

);

}
