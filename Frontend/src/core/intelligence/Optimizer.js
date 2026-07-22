export default class Optimizer{

    optimize(runtime,recommendations){

        recommendations.forEach(

            recommendation=>{

                if(runtime.apply) {
                    runtime.apply(

                        recommendation

                    );
                }

            }

        );

    }

}
