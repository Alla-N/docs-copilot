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
    sources: {
        id: number;
        title: string;
        url: string;
        score: number;
    }[];
};

export type ChatMessage = UIMessage<unknown, ChatDataParts>;
