import CommandRouter from "../lib/commands/CommandRouter";

class PluginManager{

    constructor(){

        this.plugins=[];

        this.extensions={

            providers:new Map(),

            tools:new Map(),

            artifacts:new Map(),

            commands:new Map(),

            panels:new Map(),

            shortcuts:new Map()

        };

    }

    register(plugin){

        this.plugins.push(plugin);

        plugin.activate?.({

            registerProvider:

                (id,p)=>

                this.extensions.providers.set(id,p),

            registerTool:

                (id,t)=>

                this.extensions.tools.set(id,t),

            registerArtifact:

                (id,a)=>

                this.extensions.artifacts.set(id,a),

            registerCommand:
                (command) =>
                CommandRouter.register(command),

            registerPanel:

                (id,p)=>

                this.extensions.panels.set(id,p),

            registerShortcut:

                (id,s)=>

                this.extensions.shortcuts.set(id,s)

        });

    }

    get(type){

        return this.extensions[type];

    }

}

export default new PluginManager();
