import React from "react";
import ReactDOM from "react-dom/client";

import "./styles/globals.css";
import "./styles/variables.css";
import "./styles/layout.css";
import "./styles/paper.css";
import "./styles/material.css";
import "./styles/clay.css";
import "./styles/animations.css";

import App from "./App";
import MaterialProvider from "./design/MaterialContext";
import { ChatProvider } from "./context/ChatContext";
import { WorkspaceProvider } from "./context/WorkspaceContext";
import { CommandPaletteProvider } from "./context/CommandPaletteContext";
import { SessionProvider } from "./context/SessionContext";
import { TaskProvider } from "./context/TaskContext";
import { MCPProvider } from "./context/MCPContext";
import { ThemeProvider } from "./context/ThemeContext";
import "./plugins/EchoPlugin";
import "./components/Artifacts/registerBuiltins";

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: "20px", color: "red", fontFamily: "monospace", zIndex: 9999, position: "relative", background: "white", height: "100vh" }}>
          <h2>Frontend Crashed</h2>
          <pre>{this.state.error.message}</pre>
          <pre>{this.state.error.stack}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(
    document.getElementById("root")
).render(

    <React.StrictMode>
        <ErrorBoundary>
            <ThemeProvider>
                <MaterialProvider>
                
                <ChatProvider>

                    <WorkspaceProvider>

                        <SessionProvider>
                            <TaskProvider>
                                <MCPProvider>

                                    <CommandPaletteProvider>

                                        <App />

                                    </CommandPaletteProvider>

                                </MCPProvider>
                            </TaskProvider>
                        </SessionProvider>
                    
                    </WorkspaceProvider>

                </ChatProvider>

            </MaterialProvider>
            </ThemeProvider>
        </ErrorBoundary>
    </React.StrictMode>

);