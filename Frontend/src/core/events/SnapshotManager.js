export default class SnapshotManager{

    constructor(){

        this.snapshots=[];

    }

    save(state){

        this.snapshots.push({

            id:crypto.randomUUID(),

            timestamp:Date.now(),

            state

        });

    }

    latest(){

        return this.snapshots.at(-1);

    }

}
