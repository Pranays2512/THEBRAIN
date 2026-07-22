export default class PerformanceModel{

    estimate(history){

        if(history.length===0)

            return null;

        const avg=

            history.reduce(

                (s,h)=>s+h.duration,

                0

            )/

            history.length;

        return{

            averageLatency:avg

        };

    }

}
