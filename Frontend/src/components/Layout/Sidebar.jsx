import { useState, useRef, useEffect } from "react";
import {
    Trash2,
    MessageSquare,
    Plus,
    PanelLeftClose,
    PanelLeftOpen,
    User,
    LayoutDashboard,
    Settings,
    HelpCircle,
    LogOut,
    Bell,
    Search,
    Moon,
    Sun
} from "lucide-react";

import {
    ClayButton,
    ClayIconButton
} from "../Common";
import { useChatContext } from "../../context/ChatContext";
import { useWorkspace } from "../../context/WorkspaceContext";
import { useTheme } from "../../context/ThemeContext";

import "./Sidebar.css";

export default function Sidebar({ doodlesEnabled, setDoodlesEnabled }){
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [showSettings, setShowSettings] = useState(false);
    const settingsRef = useRef(null);
    const { theme, toggleTheme } = useTheme();

    useEffect(() => {
        function handleClickOutside(event) {
            if (settingsRef.current && !settingsRef.current.contains(event.target)) {
                setShowSettings(false);
            }
        }
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);
    const {
        conversations,
        activeId,
        setActiveId,
        newConversation,
        deleteConversation
    } = useChatContext();

    const { setView } = useWorkspace();

    return(
        <aside className={`sidebar material ${isCollapsed ? 'collapsed' : ''}`}>
            
            <div className={`sidebar-restore-btn ${isCollapsed ? 'visible' : ''}`}>
                <ClayIconButton
                    size="md"
                    icon={<PanelLeftOpen />}
                    onClick={() => setIsCollapsed(false)}
                />
            </div>

            <div className="sidebar-content">
                <div className="sidebar-top">
                    <ClayIconButton
                        icon={<Plus/>}
                        onClick={() => {
                            newConversation();
                            setView('chat');
                        }}
                    />
                    <ClayIconButton
                        icon={<PanelLeftClose size={18}/>}
                        onClick={() => setIsCollapsed(true)}
                        variant="default"
                    />
                </div>

                <div className="sidebar-utilities">
                    <div className="sidebar-doodles" title="Toggle Background Doodles">
                        <span>Doodles</span>
                        <div className="on-off-switch" onClick={() => setDoodlesEnabled(!doodlesEnabled)}>
                            <div className={`switch-bg ${doodlesEnabled ? 'on' : 'off'}`} />
                            <span className={doodlesEnabled ? 'active text' : 'text'}>ON</span>
                            <span className={!doodlesEnabled ? 'active text' : 'text'}>OFF</span>
                        </div>
                    </div>
                    <div className="sidebar-action-grid">
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
                    </div>
                </div>

                    <nav className="chat-list">
                        {conversations.map(chat=>(
                            <div
                                key={chat.id}
                                className={chat.id===activeId ? "nav-item active" : "nav-item"}
                                onClick={() => {
                                    setActiveId(chat.id);
                                    setView('chat');
                                }}
                            >
                                <MessageSquare size={18}/>
                                <span>{chat.title}</span>
                                <Trash2
                                    size={16}
                                    className="delete-chat"
                                    onClick={(e)=>{
                                        e.stopPropagation();
                                        deleteConversation(chat.id);
                                    }}
                                />
                            </div>
                        ))}
                    </nav>

                    <div className="sidebar-bottom" ref={settingsRef} style={{ position: 'relative' }}>
                        {showSettings && (
                            <div className="settings-popup material">
                                <div className="settings-item">
                                    <User size={16} /> Account
                                </div>
                                <div 
                                    className="settings-item"
                                    onClick={() => {
                                        setView('dashboard');
                                        setShowSettings(false);
                                    }}
                                >
                                    <LayoutDashboard size={16} /> Dashboard
                                </div>
                                <div className="settings-item">
                                    <HelpCircle size={16} /> Help & FAQ
                                </div>
                                <div className="settings-item">
                                    <Settings size={16} /> Preferences
                                </div>
                                <div className="settings-divider"></div>
                                <div className="settings-item text-danger">
                                    <LogOut size={16} /> Log Out
                                </div>
                            </div>
                        )}
                        <ClayButton fullWidth onClick={() => setShowSettings(!showSettings)}>
                            Settings
                        </ClayButton>
                    </div>
                </div>
            </aside>
    );
}
