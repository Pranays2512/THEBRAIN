import { useState } from "react";

import {
    Check,
    Copy
} from "lucide-react";

import {
    Prism as SyntaxHighlighter
} from "react-syntax-highlighter";

import {
    oneDark,
    oneLight
} from "react-syntax-highlighter/dist/esm/styles/prism";

import "./CodeBlock.css";

export default function CodeBlock({

    language,

    children,

    embedded = false

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

        <div className={`code-wrapper ${embedded ? "code-wrapper-embedded" : ""}`}>

            {!embedded && (
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
            )}

            <SyntaxHighlighter
                language={language}
                style={embedded ? oneLight : oneDark}
                showLineNumbers
                customStyle={{
                    margin: 0,
                    borderRadius: embedded ? 0 : "16px",
                    background: embedded ? "transparent" : "rgba(0,0,0,0.03)",
                    fontSize: "15px",
                    fontFamily: "'Shantell Sans', 'Patrick Hand', 'Caveat', cursive, monospace",
                    padding: embedded ? "10px 4px 16px" : "18px",
                    minHeight: embedded ? "100%" : undefined
                }}
                codeTagProps={{
                    style: {
                        fontFamily: "'Shantell Sans', 'Patrick Hand', 'Caveat', cursive, monospace",
                        fontSize: "15px",
                        lineHeight: "1.7"
                    }
                }}
            >
                {children}
            </SyntaxHighlighter>



        </div>

    );

}
