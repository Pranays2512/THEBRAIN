import React, { useState } from 'react';
import Sidebar from "./Sidebar";
import Topbar from "./Topbar";
import Background from "./Background";
import "./AppShell.css";

export default function AppShell({ children }) {
    const [doodlesEnabled, setDoodlesEnabled] = useState(false);

    return (
        <>
            <Background enabled={doodlesEnabled} />
            <div className="app-shell">
                <Sidebar
                    doodlesEnabled={doodlesEnabled}
                    setDoodlesEnabled={setDoodlesEnabled}
                />
                <main className="app-main">
                    <section className="app-content">
                        <Topbar />
                        {children}
                    </section>
                </main>
            </div>
        </>
    );

}
