import { motion } from "framer-motion";

export default function DoodleGraph({ data, color = "var(--text)", label, detail, unit = "" }) {
    const maxVal = Math.max(...data);
    const minVal = Math.min(...data);
    // Add a small padding so lines don't hit the absolute ceiling/floor
    const padding = (maxVal - minVal) * 0.1 || (maxVal * 0.1) || 1;
    const yMax = maxVal + padding;
    const yMin = Math.max(0, minVal - padding);
    const range = yMax - yMin;
    
    const points = data.map((val, i) => {
        const x = (i / (data.length - 1)) * 100;
        const y = 100 - (((val - yMin) / range) * 100);
        return `${x},${y}`;
    }).join(" ");
    
    const currentVal = data[data.length - 1];

    return (
        <div className="doodle-graph" style={{ position: 'relative', display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                <div>
                    <div className="graph-label" style={{ marginBottom: '2px' }}>{label}</div>
                    {detail && <div style={{ fontSize: '13px', color: 'var(--muted)' }}>{detail}</div>}
                </div>
                <div style={{ fontSize: '22px', fontWeight: 'bold', color: color }}>
                    {currentVal}{unit}
                </div>
            </div>
            
            <div style={{ position: 'relative', flex: 1, display: 'flex' }}>
                {/* Y-Axis Labels */}
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', fontSize: '12px', color: 'var(--muted)', paddingRight: '12px', paddingBottom: '2px' }}>
                    <span>{yMax.toFixed(2).replace(/\.00$/, '')}{unit}</span>
                    <span>{yMin.toFixed(2).replace(/\.00$/, '')}{unit}</span>
                </div>
                
                <svg viewBox="-2 -2 104 104" preserveAspectRatio="none" style={{ flex: 1, overflow: 'visible' }}>
                    <defs>
                    <filter id={`doodle-displacement-${label.replace(/\s+/g, '')}`} colorInterpolationFilters="sRGB">
                        <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="2" result="noise" />
                        <feDisplacementMap in="SourceGraphic" in2="noise" scale="3" xChannelSelector="R" yChannelSelector="G" />
                    </filter>
                </defs>
                
                {/* Axes */}
                <polyline 
                    points="0,100 100,100" 
                    fill="none" 
                    stroke="var(--text)" 
                    strokeWidth="0.5" 
                    strokeOpacity="0.3"
                    style={{ filter: `url(#doodle-displacement-${label.replace(/\s+/g, '')})` }} 
                />
                <polyline 
                    points="0,0 0,100" 
                    fill="none" 
                    stroke="var(--text)" 
                    strokeWidth="0.5" 
                    strokeOpacity="0.3"
                    style={{ filter: `url(#doodle-displacement-${label.replace(/\s+/g, '')})` }} 
                />

                    <motion.polyline 
                        points={points} 
                        fill="none" 
                        stroke={color} 
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        style={{ filter: `url(#doodle-displacement-${label.replace(/\s+/g, '')})` }}
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: 1.5, ease: "easeOut" }}
                    />
                </svg>
            </div>
        </div>
    );
}
