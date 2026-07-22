magic_circle_code = """
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
"""

with open('src/components/Layout/Doodles.jsx', 'r') as f:
    content = f.read()

insert_pos = content.find('export const ALL_DOODLES')
new_content = content[:insert_pos] + magic_circle_code + "\n" + content[insert_pos:]

new_content = new_content.replace('archDoodle\n];', 'archDoodle,\n    magicCircleDoodle\n];')

with open('src/components/Layout/Doodles.jsx', 'w') as f:
    f.write(new_content)
