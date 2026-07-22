import {

createContext,

useContext,

useMemo

} from "react";

import TaskManager

from "../core/tasks/TaskManager";

const TaskContext=

createContext();

export function TaskProvider({

children

}){

const manager=

useMemo(

()=>new TaskManager(),

[]

);

return(

<TaskContext.Provider

value={manager}

>

{children}

</TaskContext.Provider>

);

}

export function useTasks(){

return useContext(

TaskContext

);

}
