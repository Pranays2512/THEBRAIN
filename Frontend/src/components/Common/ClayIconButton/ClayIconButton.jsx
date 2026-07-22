import clsx from "clsx";
import "./ClayIconButton.css";

export default function ClayIconButton({
    icon,
    size = "md",
    variant = "default",
    active = false,
    disabled = false,
    onClick,
    className = "",
}) {
    return (
        <button
            className={clsx(
                "clay-icon-button",
                "material",
                `icon-${size}`,
                `variant-${variant}`,
                active && "active",
                disabled && "disabled",
                className
            )}
            onClick={onClick}
            disabled={disabled}
        >
            {icon}
        </button>
    );
}
