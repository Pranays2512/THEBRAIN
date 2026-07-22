import { forwardRef } from "react";

import clsx from "clsx";

import "./ClayInput.css";

const ClayInput = forwardRef(({

    value,
    onChange,
    onKeyDown,
    placeholder = "",
    multiline = false,
    rows = 1,
    className = ""

}, ref) => {

    const Component =
        multiline ? "textarea" : "input";

    return (

        <div

            className={clsx(
                "clay-input-wrapper",
                "material",
                className
            )}

        >

            <Component

                ref={ref}

                className="clay-input"

                value={value}

                rows={rows}

                placeholder={placeholder}

                onChange={onChange}

                onKeyDown={onKeyDown}

            />

        </div>

    );

});

export default ClayInput;
