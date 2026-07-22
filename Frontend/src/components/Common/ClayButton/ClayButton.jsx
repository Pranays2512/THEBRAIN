import clsx from "clsx";
import "./ClayButton.css";

export default function ClayButton({
    children,
    icon,
    variant = "default",
    size = "md",
    fullWidth = false,
    disabled = false,
    onClick,
    className = "",
}) {
    return (
        <button
            className={clsx(
                "clay-button",
                "material",
                `variant-${variant}`,
                `size-${size}`,
                fullWidth && "full-width",
                disabled && "disabled",
                className
            )}
            onClick={onClick}
            disabled={disabled}
        >
            {icon && (
                <span className="button-icon">
                    {icon}
                </span>
            )}

            {children && (
                <span className="button-label">
                    {children}
                </span>
            )}
        </button>
    );
}
