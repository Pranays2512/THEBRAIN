export default class Principal{

    constructor({

        id,

        roles=[],

        permissions=[]

    }){

        this.id=id;

        this.roles=roles;

        this.permissions=

            new Set(

                permissions

            );

    }

    can(permission){

        return this.permissions.has(

            permission

        );

    }

}
