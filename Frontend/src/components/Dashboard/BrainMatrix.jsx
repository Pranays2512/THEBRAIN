import { useEffect, useState } from "react";
import { motion } from "framer-motion";

const NUM_NODES = 35;
const CONNECTIONS_PER_NODE = 2;

function generateNodes() {
    return Array.from({ length: NUM_NODES }).map((_, i) => ({
        id: i,
        x: Math.random() * 90 + 5,
        y: Math.random() * 90 + 5,
        size: Math.random() * 4 + 2,
    }));
}

function generateLinks(nodes) {
    const links = [];
    nodes.forEach(node => {
        for (let i = 0; i < CONNECTIONS_PER_NODE; i++) {
            const target = nodes[Math.floor(Math.random() * NUM_NODES)];
            if (target.id !== node.id) {
                links.push({ source: node, target });
            }
        }
    });
    return links;
}

export default function BrainMatrix() {
    const [nodes, setNodes] = useState([]);
    const [links, setLinks] = useState([]);

    useEffect(() => {
        const n = generateNodes();
        setNodes(n);
        setLinks(generateLinks(n));
    }, []);

    return (
        <svg style={{ width: "100%", height: "100%" }} viewBox="0 0 100 100" preserveAspectRatio="none">
            {links.map((link, i) => (
                <motion.line
                    key={i}
                    x1={`${link.source.x}%`}
                    y1={`${link.source.y}%`}
                    x2={`${link.target.x}%`}
                    y2={`${link.target.y}%`}
                    stroke="var(--text)"
                    strokeOpacity={0.15}
                    strokeWidth={0.2}
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 2, delay: Math.random() * 1.5 }}
                />
            ))}
            {nodes.map(node => (
                <motion.circle
                    key={node.id}
                    cx={`${node.x}%`}
                    cy={`${node.y}%`}
                    r={node.size / 2}
                    fill="var(--mint)"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1, opacity: [0.6, 1, 0.6] }}
                    transition={{ 
                        scale: { duration: 0.5 },
                        opacity: { repeat: Infinity, duration: 2 + Math.random() * 2 }
                    }}
                />
            ))}
        </svg>
    );
}
