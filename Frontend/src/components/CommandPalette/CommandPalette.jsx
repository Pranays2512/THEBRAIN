import { useState } from "react";
import { Command } from "cmdk";
import { useCommandPalette } from "../../context/CommandPaletteContext";
import { useChatContext } from "../../context/ChatContext";
import CommandRouter from "../../lib/commands/CommandRouter";
import fuzzySearch from "../../lib/commands/fuzzySearch";
import useRecentCommands from "../../hooks/useRecentCommands";
import "./CommandPalette.css";

export default function CommandPalette() {
    const { open, closePalette } = useCommandPalette();
    const { conversations, setActiveId } = useChatContext();
    const { recent, push } = useRecentCommands();
    const [search, setSearch] = useState("");

    const chatCommands = conversations.map(chat => ({
        id: `chat-${chat.id}`,
        title: `Go to: ${chat.title}`,
        category: "Chat",
        keywords: ["conversation"],
        run() {
            setActiveId(chat.id);
        }
    }));

    const allCommands = [
        ...CommandRouter.list(),
        ...chatCommands
    ];

    const filtered = fuzzySearch(search, allCommands);

    const grouped = filtered.reduce((acc, command) => {
        (acc[command.category] ??= []).push(command);
        return acc;
    }, {});
    
    const recentCommandsList = recent
        .map(id => allCommands.find(c => c.id === id))
        .filter(Boolean);

    function handleSelect(command) {
        command.run();
        push(command.id);
        closePalette();
    }

    return (
        <Command.Dialog
            open={open}
            onOpenChange={value => {
                if (!value) closePalette();
            }}
            label="Command Palette"
            className="command-dialog"
        >
            <Command.Input
                value={search}
                onValueChange={setSearch}
                placeholder="Search commands..."
            />
            <Command.List>
                {search === "" && recentCommandsList.length > 0 && (
                    <Command.Group heading="Recent">
                        {recentCommandsList.map(command => (
                            <Command.Item
                                key={`recent-${command.id}`}
                                onSelect={() => handleSelect(command)}
                            >
                                {command.title}
                            </Command.Item>
                        ))}
                    </Command.Group>
                )}
                {Object.entries(grouped).map(([category, list]) => (
                    <Command.Group heading={category} key={category}>
                        {list.map(command => (
                            <Command.Item
                                key={command.id}
                                onSelect={() => handleSelect(command)}
                            >
                                {command.title}
                            </Command.Item>
                        ))}
                    </Command.Group>
                ))}
            </Command.List>
        </Command.Dialog>
    );
}
