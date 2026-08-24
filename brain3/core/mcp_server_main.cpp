/**
 * brain3/core/mcp_server_main.cpp
 *
 * Driver binary for The Brain 3 Model Context Protocol (MCP) Server.
 * Supports standard stdio JSON-RPC transport and network TCP socket mode.
 */

#include "mcp_server.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    int port = 0; // 0 = stdio mode by default

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "The Brain 3 — Model Context Protocol (MCP) Server\n"
                      << "Usage:\n"
                      << "  ./brain_mcp_server             (Run in Stdio Mode for Cursor / Antigravity / Claude)\n"
                      << "  ./brain_mcp_server --port 9999 (Run in TCP Socket Server Mode for external apps)\n";
            return 0;
        }
    }

    brain3::mcp::MCPServer server;

    if (port > 0) {
        std::cout << "🚀 [TheBrain-3 MCP Server]: Starting on TCP port " << port << "...\n";
        server.start_socket_server(port, true);
    } else {
        // Stdio mode
        server.run_stdio();
    }

    return 0;
}
