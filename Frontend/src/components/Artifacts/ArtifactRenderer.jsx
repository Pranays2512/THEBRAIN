import{

getArtifact

}

from

"../../lib/artifacts/ArtifactRegistry";

export default function ArtifactRenderer({

artifact

}){

const Component=

getArtifact(

artifact.type

);

if(!Component){

return(

<div>

Unknown artifact type:

{" "}

<strong>

{artifact.type}

</strong>

</div>

);

}

return(

<Component

artifact={artifact}

/>

);

}
