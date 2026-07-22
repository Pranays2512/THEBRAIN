class MCPRegistry {

    constructor() {

        this.tools = new Map();
        this.resources = new Map();
        this.prompts = new Map();

    }

    registerTool(tool) {

        this.tools.set(tool.name, tool);

    }

    registerResource(resource) {

        this.resources.set(resource.uri, resource);

    }

    registerPrompt(prompt) {

        this.prompts.set(prompt.name, prompt);

    }

    getTools() {

        return [...this.tools.values()];

    }

    getResources() {

        return [...this.resources.values()];

    }

    getPrompts() {

        return [...this.prompts.values()];

    }

}

export default new MCPRegistry();
