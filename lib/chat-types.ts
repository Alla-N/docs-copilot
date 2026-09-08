import { UIMessage } from "ai";

// Type-only: neither module's runtime (Supabase, OpenAI clients) reaches the browser bundle.
import type { RetrievalMode } from "./retrieve";
import type { PlanIntent } from "./plan";

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
    /**
     * Which path produced this reply. Sent on every response, before any text. The UI
     * tells the reader when the reranker was unavailable ("cosine-fallback") — that path
     * refuses more and ranks worse, and saying nothing would blame the documentation.
     */
    retrieval: { mode: RetrievalMode; intent: PlanIntent };
    /**
     * HMAC over this assistant answer's text (lib/assistant-signature.ts). The client stores it
     * with the message and sends it back with the next request; the route drops any assistant
     * turn whose text does not match its signature. The value is opaque to the UI, which never
     * renders it — it exists so history can be proven server-produced rather than trusted.
     */
    signature: { sig: string };
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
