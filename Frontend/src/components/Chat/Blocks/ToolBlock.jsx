import {

Loader2,

CheckCircle2,

XCircle,

Clock

} from "lucide-react";

import "./ToolBlock.css";

export default function ToolBlock({

block

}){

const{

toolCall

}=block;

return(

<div className="tool-card">

<div className="tool-header">

<div>

{toolCall.tool}

</div>

<div>

{

toolCall.status==="running"

&&

<Loader2

size={16}

className="spin"

/>

}

{

toolCall.status==="success"

&&

<CheckCircle2

size={16}

/>

}

{

toolCall.status==="failed"

&&

<XCircle

size={16}

/>

}

{

toolCall.status==="queued"

&&

<Clock

size={16}

/>

}

</div>

</div>

<pre>

{

JSON.stringify(

toolCall.input,

null,

2

)

}

</pre>

{

toolCall.output&&

<div className="tool-output">

{toolCall.output}

</div>

}

</div>

);

}
