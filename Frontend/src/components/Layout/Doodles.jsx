import React from 'react';

export const graphDoodle = (
    <g transform="translate(100, 100)">
        <path className="sketch-line" d="M 200 200 C 300 150, 450 100, 600 250 C 750 400, 800 600, 500 700 C 200 800, 100 500, 200 200" />
        <path className="sketch-line" d="M 350 350 Q 500 200, 600 450" />
        <path className="sketch-line" d="M 600 450 Q 400 600, 350 350" />
        <path className="sketch-line" d="M 450 400 L 520 320" />
        <circle className="sketch-node" cx="200" cy="200" r="14" />
        <circle className="sketch-node" cx="600" cy="250" r="18" />
        <circle className="sketch-node" cx="500" cy="700" r="12" />
        <circle className="sketch-node" cx="350" cy="350" r="8" />
        <circle className="sketch-node" cx="600" cy="450" r="10" />
        <circle className="sketch-node" cx="450" cy="400" r="16" />
        <circle className="sketch-node" cx="520" cy="320" r="6" />
    </g>
);

export const ghostDoodle = (
    <g transform="translate(200, 100)">
        <path className="sketch-line" d="M 400 300 C 400 150, 600 150, 600 300 L 600 500 C 550 470, 500 530, 450 470 C 425 500, 400 470, 400 500 Z" />
        <circle className="sketch-node" cx="460" cy="280" r="10" />
        <circle className="sketch-node" cx="540" cy="280" r="10" />
        <path className="sketch-line" d="M 480 320 Q 500 350, 520 320" />
        <path className="sketch-line" d="M 380 350 Q 360 330, 400 320" />
        <path className="sketch-line" d="M 620 350 Q 640 330, 600 320" />
    </g>
);

export const coffeeDoodle = (
    <g transform="translate(250, 100)">
        <path className="sketch-line" d="M 350 300 L 370 500 C 370 550, 630 550, 630 500 L 650 300 Z" />
        <path className="sketch-line" d="M 640 350 C 750 350, 750 450, 635 450" />
        <path className="sketch-line" d="M 450 250 Q 420 180, 480 150 T 450 50" />
        <path className="sketch-line" d="M 550 270 Q 520 200, 580 170 T 550 70" />
        <path className="sketch-line" d="M 400 300 C 400 320, 600 320, 600 300" />
    </g>
);

export const planeDoodle = (
    <g transform="translate(200, 100)">
        <path className="sketch-line" d="M 200 600 L 700 200 L 500 700 L 400 500 L 200 600" />
        <path className="sketch-line" d="M 700 200 L 400 500" />
        <path className="sketch-line" d="M 400 500 L 450 600 L 500 700" />
        <path className="sketch-line" d="M 250 650 Q 150 750, 100 800" strokeDasharray="20, 20" />
    </g>
);

export const robotDoodle = (
    <g transform="translate(150, 100)">
        <path className="sketch-line" d="M 350 250 C 350 230, 370 230, 370 230 L 630 230 C 650 230, 650 250, 650 250 L 650 450 C 650 470, 630 470, 630 470 L 370 470 C 350 470, 350 450, 350 450 Z" />
        <circle className="sketch-node" cx="430" cy="320" r="25" />
        <circle className="sketch-node" cx="570" cy="320" r="25" />
        <circle className="sketch-node" cx="430" cy="320" r="5" />
        <circle className="sketch-node" cx="570" cy="320" r="5" />
        <path className="sketch-line" d="M 500 230 L 500 130" />
        <circle className="sketch-node" cx="500" cy="110" r="20" />
        <path className="sketch-line" d="M 430 400 L 570 400" />
        <path className="sketch-line" d="M 450 380 L 450 420" />
        <path className="sketch-line" d="M 500 380 L 500 420" />
        <path className="sketch-line" d="M 550 380 L 550 420" />
        <path className="sketch-line" d="M 350 350 L 320 350" />
        <path className="sketch-line" d="M 650 350 L 680 350" />
    </g>
);

export const rocketDoodle = (
    <g transform="translate(350, 150) scale(1.5)">
        <path className="sketch-line" d="M 100 20 C 120 50, 130 100, 130 180 L 70 180 C 70 100, 80 50, 100 20 Z" />
        <circle className="sketch-line" cx="100" cy="80" r="15" />
        <circle className="sketch-line" cx="100" cy="80" r="8" />
        <path className="sketch-line" d="M 70 140 L 40 180 L 70 180 Z" />
        <path className="sketch-line" d="M 130 140 L 160 180 L 130 180 Z" />
        <path className="sketch-line" d="M 100 140 L 100 180" />
        <path className="sketch-line" d="M 75 180 Q 80 220, 100 240 Q 120 220, 125 180" />
        <path className="sketch-line" d="M 85 180 Q 90 200, 100 210 Q 110 200, 115 180" />
        <path className="sketch-line" d="M 20 20 L 30 30 M 30 20 L 20 30" />
        <path className="sketch-line" d="M 180 60 L 190 70 M 190 60 L 180 70" />
        <path className="sketch-line" d="M 160 220 L 170 230 M 170 220 L 160 230" />
        <path className="sketch-line" d="M 30 140 L 35 145 M 35 140 L 30 145" />
    </g>
);

export const astrolabeDoodle = (
    <g transform="translate(300, 200) scale(1.2)">
        <circle className="sketch-line" cx="200" cy="200" r="150" />
        <circle className="sketch-line" cx="200" cy="200" r="140" strokeDasharray="10,10" />
        <circle className="sketch-line" cx="200" cy="200" r="80" />
        <circle className="sketch-node" cx="200" cy="200" r="10" />
        <ellipse className="sketch-line" cx="200" cy="200" rx="180" ry="60" transform="rotate(30 200 200)" />
        <ellipse className="sketch-line" cx="200" cy="200" rx="180" ry="60" transform="rotate(-45 200 200)" />
        <circle className="sketch-node" cx="80" cy="130" r="8" />
        <circle className="sketch-node" cx="350" cy="280" r="12" />
        <circle className="sketch-node" cx="120" cy="280" r="15" />
        <ellipse className="sketch-line" cx="120" cy="280" rx="25" ry="8" transform="rotate(15 120 280)" />
        <path className="sketch-line" d="M 50 200 L 350 200" strokeDasharray="5,15" />
        <path className="sketch-line" d="M 200 50 L 200 350" strokeDasharray="5,15" />
    </g>
);

export const brainDoodle = (
    <g transform="translate(350, 150) scale(1.3)">
        <path className="sketch-line" d="M 150 50 C 100 50, 50 100, 50 150 C 50 200, 80 220, 100 250 C 110 270, 130 280, 150 280 L 150 50 Z" />
        <path className="sketch-line" d="M 80 100 C 100 90, 120 110, 140 100" />
        <path className="sketch-line" d="M 60 150 C 90 140, 110 170, 140 150" />
        <path className="sketch-line" d="M 70 200 C 100 190, 120 220, 140 200" />
        <path className="sketch-line" d="M 150 50 L 200 50 L 220 70 L 220 120 L 250 150 L 250 200 L 220 230 L 200 230 L 180 250 L 180 280 L 150 280 L 150 50 Z" />
        <path className="sketch-line" d="M 150 100 L 180 100 L 190 110 L 210 110" />
        <circle className="sketch-node" cx="215" cy="110" r="4" />
        <path className="sketch-line" d="M 150 150 L 170 150 L 190 170 L 230 170" />
        <circle className="sketch-node" cx="235" cy="170" r="4" />
        <path className="sketch-line" d="M 150 200 L 180 200 L 190 190 L 200 190" />
        <circle className="sketch-node" cx="205" cy="190" r="4" />
        <path className="sketch-line" d="M 150 240 L 160 240 L 170 250" />
        <circle className="sketch-node" cx="175" cy="250" r="4" />
    </g>
);

export const cubeDoodle = (
    <g transform="translate(450, 200) scale(2)">
        <path className="sketch-line" d="M 0 50 L 86 0 L 173 50 L 173 150 L 86 200 L 0 150 Z" />
        <path className="sketch-line" d="M 86 100 L 86 200 M 86 100 L 0 50 M 86 100 L 173 50" />
        <path className="sketch-line" d="M 43 50 L 86 25 L 129 50 L 129 100 L 86 125 L 43 100 Z" strokeDasharray="5,5" />
        <path className="sketch-line" d="M 86 75 L 86 125 M 86 75 L 43 50 M 86 75 L 129 50" strokeDasharray="5,5" />
        <path className="sketch-line" d="M -50 50 L 223 50" strokeDasharray="2,4" opacity="0.5" />
        <path className="sketch-line" d="M 86 -50 L 86 250" strokeDasharray="2,4" opacity="0.5" />
        <path className="sketch-line" d="M -30 210 L 200 -20" strokeDasharray="2,4" opacity="0.5" />
    </g>
);

export const archDoodle = (
    <g transform="translate(300, 200) scale(1.5)">
        <path className="sketch-line" d="M 50 200 L 350 200" />
        <path className="sketch-line" d="M 100 200 L 100 100 L 150 50 L 200 100 L 200 200" />
        <path className="sketch-line" d="M 120 200 L 120 120 L 150 90 L 180 120 L 180 200" />
        <path className="sketch-line" d="M 220 200 L 220 120 L 280 120 L 280 200" />
        <path className="sketch-line" d="M 230 200 L 230 140 L 270 140 L 270 200" />
        <path className="sketch-line" d="M 130 200 L 130 170 Q 150 140 170 170 L 170 200" />
        <path className="sketch-line" d="M 240 200 L 240 170 Q 250 150 260 170 L 260 200" />
        <path className="sketch-line" d="M 80 150 L 300 150" strokeDasharray="3,3" />
        <path className="sketch-line" d="M 150 200 L 150 0" strokeDasharray="4,6" opacity="0.5" />
        <path className="sketch-line" d="M 320 200 L 320 50 L 250 80" />
        <path className="sketch-line" d="M 320 60 L 260 90" />
        <path className="sketch-line" d="M 250 80 L 250 120" />
        <circle className="sketch-node" cx="250" cy="120" r="3" />
    </g>
);


export const magicCircleDoodle = (
    <g transform="translate(250, 150) scale(1.3)">
        <circle className="sketch-line" cx="200" cy="200" r="150" />
        <circle className="sketch-line" cx="200" cy="200" r="140" />
        <circle className="sketch-line" cx="200" cy="200" r="130" strokeDasharray="5,10" />
        <polygon className="sketch-line" points="200,60 321,270 79,270" />
        <polygon className="sketch-line" points="200,340 79,130 321,130" />
        <circle className="sketch-line" cx="200" cy="200" r="80" />
        <polygon className="sketch-line" points="143,143 257,143 257,257 143,257" />
        <polygon className="sketch-line" points="200,120 280,200 200,280 120,200" />
        <circle className="sketch-node" cx="200" cy="200" r="15" />
        <ellipse className="sketch-line" cx="200" cy="200" rx="30" ry="15" />
        <path className="sketch-line" d="M 200 20 L 210 40 L 190 40 Z" />
        <path className="sketch-line" d="M 380 200 L 360 210 L 360 190 Z" />
        <path className="sketch-line" d="M 200 380 L 190 360 L 210 360 Z" />
        <path className="sketch-line" d="M 20 200 L 40 190 L 40 210 Z" />
        <path className="sketch-line" d="M 200 60 L 200 140" />
        <path className="sketch-line" d="M 321 270 L 257 257" />
        <path className="sketch-line" d="M 79 270 L 143 257" />
        <circle className="sketch-node" cx="200" cy="60" r="5" />
        <circle className="sketch-node" cx="321" cy="270" r="5" />
        <circle className="sketch-node" cx="79" cy="270" r="5" />
        <circle className="sketch-node" cx="200" cy="340" r="5" />
        <circle className="sketch-node" cx="79" cy="130" r="5" />
        <circle className="sketch-node" cx="321" cy="130" r="5" />
    </g>
);


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

export const ALL_DOODLES = [
    graphDoodle,
    rocketDoodle,
    ghostDoodle,
    brainDoodle,
    planeDoodle,
    astrolabeDoodle,
    robotDoodle,
    cubeDoodle,
    coffeeDoodle,
    archDoodle,
    magicCircleDoodle,
    lightbulbDoodle,
    bookDoodle,
    compassDoodle,
    hourglassDoodle,
    flaskDoodle,
    balloonDoodle
];
