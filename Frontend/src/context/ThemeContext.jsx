import React, { createContext, useContext, useEffect, useState } from "react";

const ThemeContext = createContext();

export function ThemeProvider({ children }) {
    const [theme, setTheme] = useState(() => {
        const storedTheme = localStorage.getItem("clay-theme");
        if (storedTheme) return storedTheme;
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return "dark";
        }
        return "light";
    });

    useEffect(() => {
        if (theme === "dark") {
            document.documentElement.setAttribute("data-theme", "dark");
        } else {
            document.documentElement.removeAttribute("data-theme");
        }
        localStorage.setItem("clay-theme", theme);
    }, [theme]);

    const toggleTheme = () => {
        setTheme(prev => prev === "light" ? "dark" : "light");
    };

    return (
        <ThemeContext.Provider value={{ theme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    return useContext(ThemeContext);
}
