export default function ImageArtifact({

artifact

}){

return(

<img

src={artifact.data.url}

alt={artifact.title}

style={{

width:"100%",

borderRadius:18

}}

/>

);

}
