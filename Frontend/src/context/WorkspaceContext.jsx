import {
    createContext,
    useContext,
    useMemo,
    useState,
    useCallback
} from "react";

const WorkspaceContext = createContext();

export function WorkspaceProvider({
    children
}) {
    const [artifact, setArtifact] = useState(null);
    const [currentView, setCurrentView] = useState('chat'); // 'chat' | 'dashboard'

    const openArtifact = useCallback((art) => {
        setArtifact(art);
    }, []);

    const closeArtifact = useCallback(() => {
        setArtifact(null);
    }, []);

    const value = useMemo(() => ({
        artifact,
        openArtifact,
        closeArtifact,
        currentView,
        setView: setCurrentView
    }), [artifact, openArtifact, closeArtifact, currentView]);

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
