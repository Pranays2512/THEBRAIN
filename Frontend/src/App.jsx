import AppShell from "./components/Layout/AppShell";
import Workspace from "./components/Layout/Workspace";
import CommandPalette from "./components/CommandPalette/CommandPalette";
import { useCommandPalette } from "./context/CommandPaletteContext";
import useGlobalShortcuts from "./hooks/useGlobalShortcuts";

export default function App(){
    const { togglePalette } = useCommandPalette();

    useGlobalShortcuts({
        onPalette: togglePalette
    });

    return(

        <AppShell>

            <Workspace/>
            
            <CommandPalette/>

        </AppShell>

    );

}