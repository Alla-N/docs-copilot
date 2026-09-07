import { UIMessage } from "ai";

/**
 * Custom data parts. Each key becomes a part type: "sources" -> "data-sources".
 *
 * Do NOT intersect this with UIDataTypes. UIDataTypes is Record<string, unknown>,
 * and intersecting adds a string index signature — which collapses
 * DataUIPart's `keyof DATA_TYPES & string` to plain `string`, giving every data
 * part the type `data-${string}` with `data: unknown`. The intersection meant to
 * satisfy the constraint is what destroys the narrowing. A plain object type
 * already satisfies `extends UIDataTypes` on its own.
 */
export type ChatDataParts = {
    /** One entry per PAGE (deduped by url), not per chunk — see lib/sources.ts. */
    sources: {
        id: number;
        title: string;
        url: string;
        /** Best rerank score among this page's chunks. */
        score: number;
        /** 1-based chunk numbers the prompt labelled "[Source N]" that belong to this page. */
        chunks: number[];
    }[];
};

export type ChatMessage = UIMessage<unknown, ChatDataParts>;
