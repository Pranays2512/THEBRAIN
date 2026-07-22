import {
    Code2,
    BrainCircuit,
    Palette,
    Image,
    ArrowRight
} from "lucide-react";

import { ClayButton } from "../Common";

import "./WelcomeHero.css";

const actions = [
    {
        icon: <Code2 size={18}/>,
        title: "Code",
        desc: "Build applications"
    },
    {
        icon: <BrainCircuit size={18}/>,
        title: "Research",
        desc: "Analyze ideas"
    },
    {
        icon: <Palette size={18}/>,
        title: "Design",
        desc: "Create interfaces"
    },
    {
        icon: <Image size={18}/>,
        title: "Images",
        desc: "Generate visuals"
    }
];

export default function WelcomeHero(){

    return(

        <section className="welcome">

            <div className="welcome-card material">

                <div className="welcome-text">

                    <span className="welcome-badge">

                        Premium AI Workspace

                    </span>

                    <h1>

                        Design.
                        Build.
                        Discover.

                    </h1>

                    <p>

                        A calm workspace crafted like a physical product,
                        blending industrial materials with intelligent tools.

                    </p>

                    <ClayButton
                        variant="mint"
                        icon={<ArrowRight size={18}/>}
                    >
                        Start Conversation
                    </ClayButton>

                </div>

                <div className="welcome-grid">

                    {actions.map((item)=>(
                        <div
                            className="action-card material"
                            key={item.title}
                        >

                            <div className="action-icon">

                                {item.icon}

                            </div>

                            <h3>

                                {item.title}

                            </h3>

                            <p>

                                {item.desc}

                            </p>

                        </div>
                    ))}

                </div>

            </div>

        </section>

    );

}
