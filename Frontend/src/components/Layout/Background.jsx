import React, { useState, useEffect } from 'react';
import "./Background.css";
import { ALL_DOODLES } from './Doodles';

export default function Background({ enabled = true }) {
    const [doodleIndex, setDoodleIndex] = useState(() => Math.floor(Math.random() * ALL_DOODLES.length));

    useEffect(() => {
        if (!enabled) return;
        // Cycle to the next doodle every 14 seconds randomly
        const interval = setInterval(() => {
            setDoodleIndex(prev => {
                let next = Math.floor(Math.random() * ALL_DOODLES.length);
                while (next === prev) {
                    next = Math.floor(Math.random() * ALL_DOODLES.length);
                }
                return next;
            });
        }, 14000);
        return () => clearInterval(interval);
    }, [enabled]);

    return (
        <div className="background">
            <div className="bg-grid" />
            <div className="bg-noise" />
            
            {/* key forces React to unmount and remount SVG, resetting the CSS animation */}
            {enabled && (
                <svg key={doodleIndex} className="bg-sketch" viewBox="0 0 1000 800" xmlns="http://www.w3.org/2000/svg">
                    {/* Hand-drawn ink pen filter */}
                    <defs>
                        <filter id="pen-sketch" x="-10%" y="-10%" width="120%" height="120%">
                            <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="3" result="noise" />
                            <feDisplacementMap in="SourceGraphic" in2="noise" scale="3" xChannelSelector="R" yChannelSelector="G" />
                        </filter>
                    </defs>
                    
                    {/* Wrap doodle in a group that applies the pen filter */}
                    <g filter="url(#pen-sketch)" className="doodle-group">
                        {ALL_DOODLES[doodleIndex]}
                    </g>
                </svg>
            )}
        </div>
    );
}
