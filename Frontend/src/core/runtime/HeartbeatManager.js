export default class HeartbeatManager{

    constructor(){

        this.lastSeen=new Map();

    }

    beat(workerId){

        this.lastSeen.set(

            workerId,

            Date.now()

        );

    }

    stale(timeout=30000){

        const now=Date.now();

        return [...this.lastSeen]

            .filter(

                ([,t])=>

                now-t>timeout

            )

            .map(

                ([id])=>id

            );

    }

}
