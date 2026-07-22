export default class PolicyContext {
    constructor({ principal, permission, resource, arguments: args, metadata = {} }) {
        this.principal = principal;
        this.permission = permission;
        this.resource = resource;
        this.arguments = args;
        this.metadata = metadata;
    }
}
