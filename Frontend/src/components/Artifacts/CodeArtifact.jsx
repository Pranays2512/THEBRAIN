import CodeBlock

from "../Chat/CodeBlock";

export default function CodeArtifact({

artifact

}){

return(

<CodeBlock

language={

artifact.data.language

}

embedded

>

{artifact.data.code}

</CodeBlock>

);

}
