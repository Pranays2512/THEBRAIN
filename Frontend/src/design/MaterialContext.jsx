import { createContext, useEffect, useState } from "react";

export const MaterialContext = createContext();

export default function MaterialProvider({ children }) {

    const [mouse, setMouse] = useState({
        x: window.innerWidth / 2,
        y: window.innerHeight / 2
    });

    useEffect(() => {

        const move = (e) => {

            setMouse({
                x: e.clientX,
                y: e.clientY
            });

            document.documentElement.style.setProperty(
                "--mouse-x",
                `${e.clientX}px`
            );

            document.documentElement.style.setProperty(
                "--mouse-y",
                `${e.clientY}px`
            );

        };

        window.addEventListener("mousemove", move);

        return () =>
            window.removeEventListener("mousemove", move);

    }, []);

    return (

        <MaterialContext.Provider value={mouse}>

            {children}

        </MaterialContext.Provider>

    );

}
