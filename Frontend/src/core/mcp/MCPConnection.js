export default class MCPConnection {

    constructor({

        id,

        transport

    }){

        this.id=id;

        this.transport=transport;

        this.connected=false;

    }

    async connect(){

        if (this.transport.connect) {
            await this.transport.connect();
        }

        this.connected=true;

    }

    async disconnect(){

        if (this.transport.disconnect) {
            await this.transport.disconnect();
        }

        this.connected=false;

    }

}
