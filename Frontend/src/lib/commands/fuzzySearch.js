export default function fuzzySearch(query, items) {
    if (!query) return items;

    const q = query.toLowerCase();

    return items
        .map(item => {
            const searchableText = [
                item.title,
                ...(item.aliases || []),
                ...(item.keywords || [])
            ].join(" ").toLowerCase();

            let score = 0;
            let i = 0;
            
            for (const ch of q) {
                const idx = searchableText.indexOf(ch, i);
                if (idx === -1) return null;
                
                score += idx;
                i = idx + 1;
            }

            return {
                ...item,
                score
            };
        })
        .filter(Boolean)
        .sort((a, b) => a.score - b.score);
}
