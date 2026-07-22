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
    LogOut
} from "lucide-react";

import {
    ClayButton,
    ClayIconButton
} from "../Common";
import { useChatContext } from "../../context/ChatContext";
import { useWorkspace } from "../../context/WorkspaceContext";

import "./Sidebar.css";

export default function Sidebar(){
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [showSettings, setShowSettings] = useState(false);
    const settingsRef = useRef(null);

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
