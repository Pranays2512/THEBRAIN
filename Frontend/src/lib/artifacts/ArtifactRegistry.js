const registry={};

export function registerArtifact(

type,

component

){

    registry[type]=component;

}

export function getArtifact(type){

    return registry[type];

}

export function artifactTypes(){

    return Object.keys(registry);

}
