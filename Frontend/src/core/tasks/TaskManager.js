import EventBus from "../../lib/runtime/EventBus";
import { TASK_EVENTS } from "./TaskEvents";
import { TASK_STATUS } from "./TaskTypes";

export default class TaskManager{

    constructor(){

        this.tasks=new Map();

    }

    add(task){

        this.tasks.set(task.id,task);

        EventBus.emit(

            TASK_EVENTS.CREATED,

            task

        );

        return task.id;

    }

    update(id,patch){

        const task={

            ...this.tasks.get(id),

            ...patch

        };

        this.tasks.set(id,task);

        if(patch.progress!==undefined){

            EventBus.emit(

                TASK_EVENTS.PROGRESS,

                task

            );

        }

        return task;

    }

    finish(id,result){

        return this.update(id,{

            result,

            status:TASK_STATUS.COMPLETED,

            progress:100,

            finishedAt:Date.now()

        });

    }

    fail(id,error){

        return this.update(id,{

            error,

            status:TASK_STATUS.FAILED,

            finishedAt:Date.now()

        });

    }

    cancel(id){

        return this.update(id,{

            status:TASK_STATUS.CANCELLED,

            finishedAt:Date.now()

        });

    }

    list(){

        return [...this.tasks.values()];

    }

}
