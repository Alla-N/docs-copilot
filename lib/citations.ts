/**
 * Citation markers → markdown links. Pure, dependency-free, so the renderer
 * (components/markdown.tsx) stays a thin React layer and this stays unit-testable.
 */

/** Internal href scheme for citations. No protocol, so react-markdown's url transform keeps it. */
export const CITE_PREFIX = "#source-";

/**
 * "(Source 3)", "(Sources 1 and 2)", "(Source 1, Source 4)" → one markdown link per number,
 * "[3](#source-3)", which the renderer's `a` override turns into a superscript that jumps to
 * the matching pill. Only the parenthesised form the prompt asks for is rewritten; anything
 * else stays text. Fenced code is skipped so a literal "(Source 1)" in a snippet is left alone —
 * including an unterminated fence mid-stream.
 */
const CITATION = /\((Sources?)\s+(\d+(?:\s*(?:,|and|&)\s*(?:Sources?\s+)?\d+)*)\)/gi;

export function linkCitations(text: string): string {
    return text
        .split(/(```[\s\S]*?(?:```|$))/)
        .map((part, i) =>
            i % 2 === 1
                ? part
                : part.replace(CITATION, (_m, _word, nums: string) =>
                      (nums.match(/\d+/g) ?? []).map((n) => `[${n}](${CITE_PREFIX}${n})`).join("")
                  )
        )
        .join("");
}
