import { useEffect } from "react";
import { useMouse } from "./useMouse";

export function useLighting() {
    const { x, y } = useMouse();

    useEffect(() => {
        if (x !== 0 || y !== 0) {
            document.documentElement.style.setProperty('--light-x', `${x}%`);
            document.documentElement.style.setProperty('--light-y', `${y}%`);
        }
    }, [x, y]);
}
