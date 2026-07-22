import {

FileText,

X

} from "lucide-react";

export default function AttachmentPreview({

attachment,

onRemove

}){

const image=

attachment.type.startsWith("image");

return(

<div className="attachment-card">

{

image

?

<img

src={attachment.preview}

alt="preview"

/>

:

<FileText/>

}

<div className="attachment-name">

{attachment.name}

</div>

<button

onClick={onRemove}

>

<X size={14}/>

</button>

</div>

);

}
