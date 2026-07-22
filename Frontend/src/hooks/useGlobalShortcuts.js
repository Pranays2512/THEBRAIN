import{

useEffect

}from"react";

export default function useGlobalShortcuts({

onPalette

}){

useEffect(()=>{

function handler(e){

const mod=

e.metaKey||

e.ctrlKey;

if(

mod&&

e.key==="k"

){

e.preventDefault();

onPalette();

}

}

window.addEventListener(

"keydown",

handler

);

return()=>{

window.removeEventListener(

"keydown",

handler

);

};

},[onPalette]);

}
