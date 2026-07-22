export default class IntelligenceEngine{

    constructor({

        learning,

        model,

        recommender,

        optimizer

    }){

        this.learning=learning;

        this.model=model;

        this.recommender=recommender;

        this.optimizer=optimizer;

    }

    evaluate(runtime){

        const metrics=

            this.model.estimate(

                this.learning.records

            );

        const suggestions=

            this.recommender.recommend(

                metrics

            );

        this.optimizer.optimize(

            runtime,

            suggestions

        );

    }

}
