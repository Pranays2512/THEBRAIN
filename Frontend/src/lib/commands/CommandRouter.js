class CommandRouter {

    constructor() {

        this.commands = new Map();

    }

    register(command) {

        this.commands.set(

            command.id,

            command

        );

    }

    unregister(id) {

        this.commands.delete(id);

    }

    list() {

        return [...this.commands.values()];

    }

    execute(id, args = {}) {

        const command = this.commands.get(id);

        if (!command)
            throw new Error(

                `Unknown command: ${id}`

            );

        return command.run(args);

    }

}

export default new CommandRouter();
