import Logger from "./Logger";
import Metrics from "./Metrics";
import Tracer from "./Tracer";

export default class TelemetryManager{

    constructor(){

        this.logger=new Logger();

        this.metrics=new Metrics();

        this.tracer=new Tracer();

    }

}
