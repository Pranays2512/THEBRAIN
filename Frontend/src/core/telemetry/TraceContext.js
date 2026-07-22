export default class TraceContext {
    constructor() {
        this.traceId = crypto.randomUUID();
    }
}
