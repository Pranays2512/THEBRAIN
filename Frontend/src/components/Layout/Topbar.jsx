import { useState } from "react";
import { Bell, Search, User, Moon, Sun, ChevronUp, ChevronDown } from "lucide-react";
import { ClayIconButton } from "../Common";
import { useTheme } from "../../context/ThemeContext";
import "./Topbar.css";

export default function Topbar({ doodlesEnabled, setDoodlesEnabled }){
    const { theme, toggleTheme } = useTheme();
    const [isCollapsed, setIsCollapsed] = useState(false);

    return(
        <>
            <header className={`topbar ${isCollapsed ? 'collapsed' : ''}`}>
                <div>
                    <h1>Brain</h1>
                </div>
                <div className="topbar-actions">
                    <div className="on-off-container" title="Toggle Background Doodles">
                        <span style={{ fontSize: '14px', fontWeight: 500, color: 'var(--muted)', userSelect: 'none', marginRight: '8px' }}>Doodles</span>
                        <div className="on-off-switch" onClick={() => setDoodlesEnabled(!doodlesEnabled)}>
                            <div className={`switch-bg ${doodlesEnabled ? 'on' : 'off'}`} />
                            <span className={doodlesEnabled ? 'active text' : 'text'}>ON</span>
                            <span className={!doodlesEnabled ? 'active text' : 'text'}>OFF</span>
                        </div>
                    </div>
                    <div className="separator" />
                    <ClayIconButton
                        size="sm"
                        icon={theme === 'light' ? <Moon/> : <Sun/>}
                        onClick={toggleTheme}
                    />
                    <ClayIconButton
                        size="sm"
                        icon={<Search/>}
                    />
                    <ClayIconButton
                        size="sm"
                        icon={<Bell/>}
                    />
                    <ClayIconButton
                        size="sm"
                        variant="mint"
                        icon={<User/>}
                    />
                    <div className="separator" />
                    <ClayIconButton
                        size="sm"
                        icon={<ChevronUp/>}
                        onClick={() => setIsCollapsed(true)}
                    />
                </div>
            </header>

            {isCollapsed && (
                <div className="topbar-restore">
                    <ClayIconButton
                        size="sm"
                        icon={<ChevronDown/>}
                        onClick={() => setIsCollapsed(false)}
                    />
                </div>
            )}
        </>
    );
}
