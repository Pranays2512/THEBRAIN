export default class SecretManager {
    constructor() {
        this.vault = new Map();
    }
    
    set(key, value) {
        this.vault.set(key, value);
    }
    
    get(key) {
        return this.vault.get(key);
    }
}
