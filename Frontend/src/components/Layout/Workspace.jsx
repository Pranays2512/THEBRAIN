import { Group, Panel, Separator } from "react-resizable-panels";
import ChatArea from "./ChatArea";
import Dashboard from "../Dashboard/Dashboard";
import ArtifactPanel from "./ArtifactPanel";
import { useWorkspace } from "../../context/WorkspaceContext";
import "./Workspace.css";

export default function Workspace() {
    const { artifact, currentView } = useWorkspace();

    return (
        <Group
            key={artifact ? "with-artifact" : "without-artifact"}
            direction="horizontal"
            style={{ flex: 1, minHeight: 0, width: "100%", height: "100%" }}
        >
            <Panel defaultSize={artifact ? 64 : 100} minSize="360px">
                {currentView === 'dashboard' ? (
                    <Dashboard />
                ) : (
                    <ChatArea />
                )}
            </Panel>

            {artifact && (
                <>
                    <Separator className="resize-handle" />
                    <Panel defaultSize={36} minSize="320px" maxSize="70%">
                        <ArtifactPanel />
                    </Panel>
                </>
            )}
        </Group>
    );
}

