export default class FeedbackEngine {
    constructor(store) {
        this.store = store;
    }
    record(event) {
        this.store.add({
            ...event,
            source: "feedback"
        });
    }
}
