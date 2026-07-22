import {

Copy,

RotateCcw,

Pencil,

ThumbsUp,

ThumbsDown

} from "lucide-react";

import { ClayIconButton } from "../Common";

export default function MessageToolbar({

onCopy,

onRetry,

onEdit

}){

return(

<div className="message-toolbar">

<ClayIconButton

icon={<Copy size={15}/>}

onClick={onCopy}

/>

<ClayIconButton

icon={<RotateCcw size={15}/>}

onClick={onRetry}

/>

<ClayIconButton

icon={<Pencil size={15}/>}

onClick={onEdit}

/>

<ClayIconButton

icon={<ThumbsUp size={15}/>}

/>

<ClayIconButton

icon={<ThumbsDown size={15}/>}

/>

</div>

);

}
