import{

X

}

from"lucide-react";

import{

useWorkspace

}

from

"../../context/WorkspaceContext";

import ArtifactRenderer from "../Artifacts/ArtifactRenderer";

export default function ArtifactPanel(){

const{

artifact,

closeArtifact

}=useWorkspace();

if(!artifact)

return null;

return(

<div className="artifact-panel">

<div className="artifact-header">

<h3>

{artifact.title}

</h3>

<button

onClick={

closeArtifact

}

>

<X/>

</button>

</div>

<div

className="artifact-content"

>

<ArtifactRenderer

artifact={artifact}

/>

</div>

</div>

);

}
