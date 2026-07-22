import { Group, Panel, Separator } from "react-resizable-panels";

import ChatArea from "./ChatArea";
import Dashboard from "../Dashboard/Dashboard";
import ArtifactPanel from "./ArtifactPanel";
import { useWorkspace } from "../../context/WorkspaceContext";
import "./Workspace.css";

export default function Workspace(){
    const { artifact, currentView } = useWorkspace();

return(

<Group
direction="horizontal"
style={{ flex: 1, minHeight: 0 }}
>

<Panel
    defaultSize={artifact ? 60 : 100}
    minSize={40}
>
    {currentView === 'dashboard' ? (
        <Dashboard />
    ) : (
        <ChatArea />
    )}
</Panel>

{

artifact&&

<>

<Separator

className="resize-handle"

/>

<Panel

defaultSize={40}

minSize={25}

>

<ArtifactPanel/>

</Panel>

</>

}

</Group>

);

}
