import clsx from "clsx";
import "./ClaySurface.css";

export default function ClaySurface({
    children,
    className = "",
    hover = true,
    depth = "normal",
    onClick
}) {

    return (
        <div
            className={clsx(
                "clay",
                "material",
                `depth-${depth}`,
                hover && "hoverable",
                className
            )}
            onClick={onClick}
        >
            <div className="clay-highlight"></div>

            <div className="clay-noise"></div>

            <div className="clay-inner">

                {children}

            </div>

        </div>
    );

}
