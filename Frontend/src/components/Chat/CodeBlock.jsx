import { useState } from "react";

import {
    Check,
    Copy
} from "lucide-react";

import {
    Prism as SyntaxHighlighter
} from "react-syntax-highlighter";

import {
    oneDark
} from "react-syntax-highlighter/dist/esm/styles/prism";

import "./CodeBlock.css";

export default function CodeBlock({

    language,

    children

}){

    const [copied,setCopied]=useState(false);

    async function copy(){

        await navigator.clipboard.writeText(children);

        setCopied(true);

        setTimeout(()=>{

            setCopied(false);

        },1500);

    }

    return(

        <div className="code-wrapper">

            <div className="code-header">

                <span>

                    {language.toUpperCase()}

                </span>

                <button

                    className="copy-button"

                    onClick={copy}

                >

                    {

                        copied

                        ?

                        <Check size={16}/>

                        :

                        <Copy size={16}/>

                    }

                </button>

            </div>

            <SyntaxHighlighter

                language={language}

                style={oneDark}

                showLineNumbers

                customStyle={{

                    margin:0,

                    borderRadius:0,

                    background:"transparent",

                    fontSize:"14px",

                    padding:"22px"

                }}

            >

                {children}

            </SyntaxHighlighter>

        </div>

    );

}
