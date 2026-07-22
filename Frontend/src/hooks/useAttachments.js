import { useState } from "react";

export default function useAttachments(){

    const [attachments,setAttachments]=useState([]);

    function addFiles(files){

        const mapped=[...files].map(file=>({

            id:crypto.randomUUID(),

            file,

            name:file.name,

            size:file.size,

            type:file.type,

            preview:

                file.type.startsWith("image")

                ? URL.createObjectURL(file)

                : null

        }));

        setAttachments(prev=>[

            ...prev,

            ...mapped

        ]);

    }

    function removeAttachment(id){

        setAttachments(prev=>

            prev.filter(

                a=>a.id!==id

            )

        );

    }

    function clear(){

        attachments.forEach(a=>{

            if(a.preview)

                URL.revokeObjectURL(

                    a.preview

                );

        });

        setAttachments([]);

    }

    return{

        attachments,

        addFiles,

        removeAttachment,

        clear

    };

}
