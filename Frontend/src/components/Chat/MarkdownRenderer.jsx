import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
    Prism as SyntaxHighlighter
} from "react-syntax-highlighter";

import {
    oneDark
} from "react-syntax-highlighter/dist/esm/styles/prism";

import CodeBlock from "./CodeBlock";

import "./MarkdownRenderer.css";

export default function MarkdownRenderer({ children }) {
    return (
        <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
                p(props) {
                    return <p className="md-p" {...props} />;
                },

                h1(props) {
                    return <h1 className="md-h1" {...props} />;
                },

                h2(props) {
                    return <h2 className="md-h2" {...props} />;
                },

                h3(props) {
                    return <h3 className="md-h3" {...props} />;
                },

                ul(props) {
                    return <ul className="md-ul" {...props} />;
                },

                ol(props) {
                    return <ol className="md-ol" {...props} />;
                },

                li(props) {
                    return <li className="md-li" {...props} />;
                },

                blockquote(props) {
                    return (
                        <blockquote
                            className="md-quote"
                            {...props}
                        />
                    );
                },

                code({
                    inline,
                    className,
                    children
                }){

                    const language =
                        className?.replace("language-","") || "text";

                    if(inline){

                        return(

                            <code className="inline-code">

                                {children}

                            </code>

                        );

                    }

                    return(

                        <CodeBlock
                            language={language}
                        >

                            {String(children).replace(/\n$/,"")}

                        </CodeBlock>

                    );

                }

            }}
        >
            {children}
        </ReactMarkdown>
    );
}
