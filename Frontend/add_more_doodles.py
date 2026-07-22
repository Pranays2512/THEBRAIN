more_doodles_code = """
export const lightbulbDoodle = (
    <g transform="translate(300, 150) scale(1.5)">
        <path className="sketch-line" d="M 200 100 C 150 100, 150 180, 180 220 L 180 260 L 220 260 L 220 220 C 250 180, 250 100, 200 100 Z" />
        <path className="sketch-line" d="M 170 270 L 230 270" />
        <path className="sketch-line" d="M 175 280 L 225 280" />
        <path className="sketch-line" d="M 180 290 L 220 290" />
        <path className="sketch-line" d="M 190 300 Q 200 320, 210 300" />
        <path className="sketch-line" d="M 190 260 L 190 180 L 195 170 L 200 180 L 205 170 L 210 180 L 210 260" />
        <path className="sketch-line" d="M 180 180 L 220 180" strokeDasharray="3,3" />
        <path className="sketch-line" d="M 200 80 L 200 50" />
        <path className="sketch-line" d="M 150 70 L 120 40" />
        <path className="sketch-line" d="M 250 70 L 280 40" />
        <path className="sketch-line" d="M 130 140 L 90 140" />
        <path className="sketch-line" d="M 270 140 L 310 140" />
        <path className="sketch-line" d="M 140 200 L 110 220" />
        <path className="sketch-line" d="M 260 200 L 290 220" />
        <circle className="sketch-node" cx="200" cy="50" r="3" />
        <circle className="sketch-node" cx="120" cy="40" r="3" />
        <circle className="sketch-node" cx="280" cy="40" r="3" />
    </g>
);

export const bookDoodle = (
    <g transform="translate(250, 250) scale(1.5)">
        <path className="sketch-line" d="M 200 180 C 150 170, 100 180, 50 200 L 50 100 C 100 80, 150 70, 200 100 Z" />
        <path className="sketch-line" d="M 200 180 C 250 170, 300 180, 350 200 L 350 100 C 300 80, 250 70, 200 100 Z" />
        <path className="sketch-line" d="M 200 100 L 200 180" />
        <path className="sketch-line" d="M 50 110 C 100 90, 150 80, 200 110" />
        <path className="sketch-line" d="M 50 120 C 100 100, 150 90, 200 120" />
        <path className="sketch-line" d="M 200 110 C 250 80, 300 90, 350 110" />
        <path className="sketch-line" d="M 200 120 C 250 90, 300 100, 350 120" />
        <path className="sketch-line" d="M 70 130 C 110 120, 150 125, 180 140" strokeDasharray="5,10" />
        <path className="sketch-line" d="M 70 150 C 110 140, 150 145, 180 160" strokeDasharray="10,10" />
        <path className="sketch-line" d="M 70 170 C 110 160, 150 165, 180 180" strokeDasharray="5,5" />
        <path className="sketch-line" d="M 220 140 C 250 125, 290 120, 330 130" strokeDasharray="8,8" />
        <path className="sketch-line" d="M 220 160 C 250 145, 290 140, 330 150" strokeDasharray="4,8" />
        <path className="sketch-line" d="M 220 180 C 250 165, 290 160, 330 170" strokeDasharray="6,6" />
        <circle className="sketch-node" cx="200" cy="180" r="5" />
    </g>
);

export const compassDoodle = (
    <g transform="translate(300, 200) scale(1.3)">
        <circle className="sketch-line" cx="200" cy="200" r="100" />
        <circle className="sketch-line" cx="200" cy="200" r="120" />
        <circle className="sketch-line" cx="200" cy="200" r="90" strokeDasharray="5,5" />
        <polygon className="sketch-line" points="200,80 220,180 200,200 180,180" />
        <polygon className="sketch-line" points="200,320 220,220 200,200 180,220" />
        <polygon className="sketch-line" points="80,200 180,180 200,200 180,220" />
        <polygon className="sketch-line" points="320,200 220,180 200,200 220,220" />
        <polygon className="sketch-line" points="200,80 200,320" />
        <polygon className="sketch-line" points="80,200 320,200" />
        <circle className="sketch-node" cx="200" cy="200" r="15" />
        <path className="sketch-line" d="M 120 120 L 175 175 M 280 120 L 225 175 M 120 280 L 175 225 M 280 280 L 225 225" />
        <path className="sketch-line" d="M 190 50 L 190 30 L 210 50 L 210 30" />
        <path className="sketch-line" d="M 350 190 L 370 190 M 350 200 L 360 200 M 350 210 L 370 210 M 350 190 L 350 210" />
    </g>
);

export const hourglassDoodle = (
    <g transform="translate(350, 150) scale(1.5)">
        <path className="sketch-line" d="M 100 50 L 300 50 M 110 60 L 290 60" />
        <path className="sketch-line" d="M 100 350 L 300 350 M 110 340 L 290 340" />
        <path className="sketch-line" d="M 120 60 C 120 150, 180 180, 190 200 C 180 220, 120 250, 120 340" />
        <path className="sketch-line" d="M 280 60 C 280 150, 220 180, 210 200 C 220 220, 280 250, 280 340" />
        <ellipse className="sketch-line" cx="200" cy="200" rx="15" ry="5" />
        <path className="sketch-line" d="M 110 50 L 110 350" />
        <path className="sketch-line" d="M 290 50 L 290 350" />
        <path className="sketch-line" d="M 130 100 C 170 110, 230 110, 270 100 C 270 140, 220 160, 205 185 L 195 185 C 180 160, 130 140, 130 100 Z" />
        <path className="sketch-line" d="M 140 120 L 260 120 M 150 140 L 250 140" strokeDasharray="4,6" />
        <path className="sketch-line" d="M 200 190 L 200 290" strokeDasharray="3,3" />
        <path className="sketch-line" d="M 150 340 Q 200 270, 250 340" />
        <path className="sketch-line" d="M 160 340 Q 200 290, 240 340" />
        <path className="sketch-line" d="M 170 340 Q 200 300, 230 340" />
        <circle className="sketch-node" cx="200" cy="280" r="2" />
        <circle className="sketch-node" cx="205" cy="275" r="1.5" />
    </g>
);

export const flaskDoodle = (
    <g transform="translate(350, 200) scale(1.5)">
        <path className="sketch-line" d="M 180 50 L 220 50 L 220 150 L 280 280 C 290 300, 270 330, 200 330 C 130 330, 110 300, 120 280 L 180 150 Z" />
        <ellipse className="sketch-line" cx="200" cy="50" rx="20" ry="5" />
        <path className="sketch-line" d="M 135 240 Q 170 230, 200 240 Q 230 250, 265 240" />
        <path className="sketch-line" d="M 130 255 Q 200 280, 270 255" strokeDasharray="5,10" />
        <path className="sketch-line" d="M 140 275 Q 200 290, 260 275" strokeDasharray="5,10" />
        <circle className="sketch-line" cx="180" cy="270" r="8" />
        <circle className="sketch-line" cx="230" cy="290" r="5" />
        <circle className="sketch-line" cx="200" cy="300" r="10" />
        <circle className="sketch-line" cx="190" cy="210" r="6" />
        <circle className="sketch-line" cx="210" cy="180" r="4" />
        <circle className="sketch-line" cx="200" cy="120" r="3" />
        <circle className="sketch-line" cx="195" cy="80" r="5" />
        <path className="sketch-line" d="M 175 100 L 185 100 M 170 120 L 185 120 M 175 140 L 185 140 M 165 170 L 180 170" />
        <path className="sketch-line" d="M 135 280 C 135 300, 150 315, 170 320" />
    </g>
);

export const balloonDoodle = (
    <g transform="translate(300, 100) scale(1.3)">
        <path className="sketch-line" d="M 200 50 C 100 50, 80 150, 120 220 L 160 300 L 240 300 L 280 220 C 320 150, 300 50, 200 50 Z" />
        <path className="sketch-line" d="M 120 220 C 150 240, 250 240, 280 220" />
        <path className="sketch-line" d="M 105 150 C 150 170, 250 170, 295 150" />
        <path className="sketch-line" d="M 200 50 C 200 150, 200 250, 200 300" />
        <path className="sketch-line" d="M 200 50 C 150 150, 150 250, 160 300" />
        <path className="sketch-line" d="M 200 50 C 250 150, 250 250, 240 300" />
        <path className="sketch-line" d="M 170 350 L 230 350 L 220 400 L 180 400 Z" />
        <path className="sketch-line" d="M 170 350 L 180 400 M 180 350 L 190 400 M 190 350 L 200 400 M 200 350 L 210 400 M 210 350 L 220 400 M 220 350 L 230 400" />
        <path className="sketch-line" d="M 175 365 L 225 365 M 175 380 L 225 380" />
        <path className="sketch-line" d="M 160 300 L 170 350" />
        <path className="sketch-line" d="M 240 300 L 230 350" />
        <path className="sketch-line" d="M 185 300 L 190 350" />
        <path className="sketch-line" d="M 215 300 L 210 350" />
        <path className="sketch-line" d="M 50 100 Q 70 80, 100 100 Q 120 90, 130 110 Q 140 130, 110 140 Q 70 150, 50 130 Q 30 110, 50 100" />
        <path className="sketch-line" d="M 350 200 Q 370 180, 400 200 Q 420 190, 430 210 Q 440 230, 410 240 Q 370 250, 350 230 Q 330 210, 350 200" />
    </g>
);
"""

with open('src/components/Layout/Doodles.jsx', 'r') as f:
    content = f.read()

insert_pos = content.find('export const ALL_DOODLES')
new_content = content[:insert_pos] + more_doodles_code + "\n" + content[insert_pos:]

replacement = """archDoodle,
    magicCircleDoodle,
    lightbulbDoodle,
    bookDoodle,
    compassDoodle,
    hourglassDoodle,
    flaskDoodle,
    balloonDoodle
];"""
new_content = new_content.replace('archDoodle,\n    magicCircleDoodle\n];', replacement)

with open('src/components/Layout/Doodles.jsx', 'w') as f:
    f.write(new_content)
