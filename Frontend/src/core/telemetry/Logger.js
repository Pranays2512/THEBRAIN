export default class Logger{

    info(message,data={}){

        console.info({

            level:"info",

            timestamp:Date.now(),

            message,

            data

        });

    }

    warn(message,data={}){

        console.warn({

            level:"warn",

            timestamp:Date.now(),

            message,

            data

        });

    }

    error(message,data={}){

        console.error({

            level:"error",

            timestamp:Date.now(),

            message,

            data

        });

    }

}
