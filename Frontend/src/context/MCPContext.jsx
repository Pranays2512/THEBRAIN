import {

createContext,

useContext,

useMemo

} from "react";

import MCPClient

from "../core/mcp/MCPClient";

const MCPContext=

createContext();

export function MCPProvider({

children

}){

const client=

useMemo(

()=>new MCPClient(),

[]

);

return(

<MCPContext.Provider

value={client}

>

{children}

</MCPContext.Provider>

);

}

export function useMCP(){

return useContext(

MCPContext

);

}
