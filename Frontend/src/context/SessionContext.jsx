import {

createContext,

useContext,

useMemo

} from "react";

import SessionEngine

from "../core/session/SessionEngine";

import providers

from "../providers/providerRegistry";

const SessionContext=

createContext();

export function SessionProvider({

children

}){

const engine=

useMemo(

()=>new SessionEngine({

provider:

providers.fastapi

}),

[]

);

return(

<SessionContext.Provider

value={engine}

>

{children}

</SessionContext.Provider>

);

}

export function useSession(){

return useContext(

SessionContext

);

}
