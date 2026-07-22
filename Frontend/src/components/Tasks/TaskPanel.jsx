import { useTasks } from "../../context/TaskContext";
import { useEffect, useState } from "react";
import EventBus from "../../lib/runtime/EventBus";
import { TASK_EVENTS } from "../../core/tasks/TaskEvents";
import { TASK_STATUS } from "../../core/tasks/TaskTypes";
import TaskCard from "./TaskCard";

export default function TaskPanel() {
    const taskManager = useTasks();
    const [tasks, setTasks] = useState(() => taskManager.list());

    useEffect(() => {
        function updateTasks() {
            setTasks([...taskManager.list()]);
        }

        const unsubs = [
            EventBus.subscribe(TASK_EVENTS.CREATED, updateTasks),
            EventBus.subscribe(TASK_EVENTS.PROGRESS, updateTasks),
            EventBus.subscribe(TASK_EVENTS.COMPLETED, updateTasks),
            EventBus.subscribe(TASK_EVENTS.FAILED, updateTasks),
            EventBus.subscribe(TASK_EVENTS.CANCELLED, updateTasks),
        ];

        return () => {
            unsubs.forEach(u => u());
        };
    }, [taskManager]);

    const running = tasks.filter(t => t.status === TASK_STATUS.RUNNING);
    const queued = tasks.filter(t => t.status === TASK_STATUS.QUEUED);
    const completed = tasks.filter(t => t.status === TASK_STATUS.COMPLETED || t.status === TASK_STATUS.FAILED || t.status === TASK_STATUS.CANCELLED);

    return (
        <div style={{ padding: 18, background: "white", borderLeft: "1px solid rgba(0,0,0,0.08)", height: "100%", overflowY: "auto" }}>
            {running.length > 0 && (
                <>
                    <h4 style={{ margin: "0 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Running</h4>
                    <hr style={{ border: "none", borderTop: "1px solid #eaeaea", marginBottom: 12 }} />
                    {running.map(t => <TaskCard key={t.id} task={t} />)}
                </>
            )}

            {queued.length > 0 && (
                <>
                    <h4 style={{ margin: "20px 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Queued</h4>
                    <hr style={{ border: "none", borderTop: "1px solid #eaeaea", marginBottom: 12 }} />
                    {queued.map(t => <TaskCard key={t.id} task={t} />)}
                </>
            )}

            {completed.length > 0 && (
                <>
                    <h4 style={{ margin: "20px 0 10px 0", fontSize: 13, textTransform: "uppercase", color: "#888" }}>Completed</h4>
                    <hr style={{ border: "none", borderTop: "1px solid #eaeaea", marginBottom: 12 }} />
                    {completed.map(t => <TaskCard key={t.id} task={t} />)}
                </>
            )}
            
            {tasks.length === 0 && <div style={{ color: "#888", fontSize: 14 }}>No active tasks.</div>}
        </div>
    );
}
