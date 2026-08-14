import {

Copy,

RotateCcw

} from "lucide-react";

import { ClayIconButton } from "../Common";

export default function MessageToolbar({

onCopy,

onRetry,

retryDisabled = false

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

disabled={retryDisabled}

/>

</div>

);

}
