export default class LeaseManager{

    constructor(){

        this.leases=new Map();

    }

    acquire(

        workerId,

        jobId

    ){

        this.leases.set(

            jobId,

            workerId

        );

    }

    release(jobId){

        this.leases.delete(jobId);

    }

}
