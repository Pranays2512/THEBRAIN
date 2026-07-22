import {

createContext,

useContext

} from "react";

import PluginManager

from "./PluginManager";

const PluginContext=

createContext(

PluginManager

);

export function usePlugins(){

    return useContext(

        PluginContext

    );

}
