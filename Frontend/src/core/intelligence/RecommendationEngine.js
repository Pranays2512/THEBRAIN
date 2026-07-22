export default class RecommendationEngine{

    recommend(metrics){

        const recommendations=[];

        if(metrics && metrics.averageLatency>2000){

            recommendations.push({

                type:"provider",

                message:

                "Switch to faster provider"

            });

        }

        return recommendations;

    }

}
