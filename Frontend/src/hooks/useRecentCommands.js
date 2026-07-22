import { useState } from "react";

export default function useRecentCommands(){

const[recent,setRecent]=

useState([]);

function push(id){

setRecent(prev=>[

id,

...prev.filter(

v=>v!==id

)

].slice(0,10));

}

return{

recent,

push

};

}
