import {
    createContext,
    useContext,
    useMemo,
    useState
} from "react";

const WorkspaceContext=createContext();

export function WorkspaceProvider({
    children
}){
    const [artifact, setArtifact] = useState(null);
    const [currentView, setCurrentView] = useState('chat'); // 'chat' | 'dashboard'

    const value = useMemo(() => ({
        artifact,
        openArtifact: setArtifact,
        closeArtifact() {
            setArtifact(null);
        },
        currentView,
        setView: setCurrentView
    }), [artifact, currentView]);

    return (
        <WorkspaceContext.Provider value={value}>
            {children}
        </WorkspaceContext.Provider>
    );
}

export function useWorkspace(){

    return useContext(

        WorkspaceContext

    );

}
