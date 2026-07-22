import {

FileText

} from "lucide-react";

export default function FileBlock({

block

}){

return(

<div className="file-card">

<FileText size={18}/>

<div>

<strong>

{block.name}

</strong>

<div>

{block.mime}

</div>

</div>

</div>

);

}
