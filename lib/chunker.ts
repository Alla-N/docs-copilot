function pushCapped(chunks: string[], text: string, maxLen: number) {
    const t = text.trim();
    if (!t) return;
    if (t.length <= maxLen) {
        chunks.push(t);
        return;
    }
    for (let i = 0; i < t.length; i += maxLen) {
        chunks.push(t.slice(i, i + maxLen));
    }
}

export function chunkPage(markdown: string, maxLen = 1500): string[] {
    const sections = markdown.split(/\n(?=#{1,3} )/);
    const chunks: string[] = [];

    for (const raw of sections) {
        const section = raw.trim();
        if (!section) continue;

        if (section.length <= maxLen) {
            pushCapped(chunks, section, maxLen);
            continue;
        }

        const heading = section.match(/^#{1,3} .+/)?.[0] ?? "";
        let current = "";
        for (const p of section.split(/\n\n+/)) {
            const candidate = current ? current + "\n\n" + p : p;
            if (candidate.length > maxLen && current) {
                pushCapped(chunks, current, maxLen);
                current = heading ? heading + "\n\n" + p : p; // heading rides along
            } else {
                current = candidate;
            }
        }
        if (current.trim()) {
            // Hard-split anything that still exceeds the cap (no paragraph breaks to use)
            if (current.length > maxLen) {
                for (let i = 0; i < current.length; i += maxLen) {
                    pushCapped(chunks, current.slice(i, i + maxLen), maxLen);
                }
            } else {
                pushCapped(chunks, current, maxLen);
            }
        }
    }

    // Drop near-empty fragments — they pollute retrieval with noise vectors
    return chunks.filter((c) => c.length > 80);
}

export function stripBoilerplate(markdown: string): string {
    // Remove the "## Navigation" section and everything until the next h2
    return markdown.replace(/\n## Navigation[\s\S]*?(?=\n## |$)/g, "\n");
}
