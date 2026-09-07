"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Markdown } from "@/components/markdown";
import { ChatMessage } from "@/lib/chat-types";
import { captureLanding, landingHeaders } from "@/lib/landing";
import { isRefusal } from "@/lib/refusal";
import { parseRateLimit, REPO_URL } from "@/lib/rate-limit-message";

/**
 * Empty-state prompts. Drawn from evals/manual-qa.md (groups 1 and 2) so each is a question
 * the corpus is known to answer — a suggested question that refuses would be a bad first
 * impression. The last one is deliberately OUT of scope: refusing is the project's central
 * claim, and a visitor should be able to see it happen without having to think of a trap.
 */
const SUGGESTIONS: { text: string; hint?: string }[] = [
    { text: "How do I stream text with streamText?" },
    { text: "How do I generate structured JSON with a Zod schema?" },
    { text: "What is tool calling?" },
    { text: "What's new in AI SDK 7?" },
    { text: "embeddings?", hint: "terse on purpose" },
    { text: "How do I fine-tune a model with the AI SDK?", hint: "out of scope — watch it refuse" },
];

const CORPUS_URL = `${REPO_URL}/blob/main/scripts/ingest.ts`;

/** Pixels from the bottom within which the reader counts as "following along". */
const STICK_THRESHOLD = 80;

export default function Chat() {
    const [input, setInput] = useState("");

    // Where did this visitor come from? Captured once per session on mount, then sent as
    // headers with every chat request so the query log can attribute questions to a channel.
    // `headers` is resolved per request, so sessionStorage is read at send time — never on
    // the server render. See specs/visitor-analytics.md.
    useEffect(() => captureLanding(), []);
    const transport = useMemo(
        () => new DefaultChatTransport<ChatMessage>({ api: "/api/chat", headers: landingHeaders }),
        []
    );
    const { messages, sendMessage, status, error, regenerate, stop } = useChat<ChatMessage>({ transport });

    const isBusy = status === "submitted" || status === "streaming";
    const rateLimit = parseRateLimit(error);

    // ── Layout: the input bar is fixed, so the list needs exactly its height as bottom
    // padding — a guessed constant was too small on some viewports and hid the last lines
    // of an answer under the bar. Measured live; ResizeObserver covers font/width changes.
    const barRef = useRef<HTMLDivElement>(null);
    const [barHeight, setBarHeight] = useState(176);
    useEffect(() => {
        const el = barRef.current;
        if (!el) return;
        const update = () => setBarHeight(el.offsetHeight);
        update();
        const ro = new ResizeObserver(update);
        ro.observe(el);
        return () => ro.disconnect();
    }, []);

    // ── Auto-scroll: follow the stream while the reader is at the bottom; the moment they
    // scroll up to re-read something, stop yanking the page. "At the bottom" is re-evaluated
    // on every scroll, so scrolling back down re-engages following.
    const stickRef = useRef(true);
    useEffect(() => {
        const onScroll = () => {
            const distance = document.documentElement.scrollHeight - window.innerHeight - window.scrollY;
            stickRef.current = distance < STICK_THRESHOLD;
        };
        window.addEventListener("scroll", onScroll, { passive: true });
        return () => window.removeEventListener("scroll", onScroll);
    }, []);
    useEffect(() => {
        if (stickRef.current) window.scrollTo({ top: document.documentElement.scrollHeight });
    }, [messages, status]);

    const ask = useCallback(
        (text: string) => {
            if (!text.trim() || isBusy) return;
            stickRef.current = true; // a new question always starts at the bottom
            sendMessage({ text });
            setInput("");
        },
        [isBusy, sendMessage]
    );

    return (
        <div className="flex flex-col w-full max-w-3xl mx-auto px-4 pt-10" style={{ paddingBottom: barHeight + 24 }}>
            <header className="mb-8">
                <h1 className="text-lg font-semibold tracking-tight">docs-copilot</h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                    Grounded answers from the Vercel AI SDK documentation — with sources, or a refusal.
                </p>
            </header>

            {messages.length === 0 && !error && (
                // Empty state. A first-time visitor (often a recruiter) sees what to type, what the
                // thing knows, and where the code is — same voice as the rate-limit banner.
                <section className="space-y-5">
                    <div className="flex flex-wrap gap-2">
                        {SUGGESTIONS.map((s) => (
                            <button
                                key={s.text}
                                type="button"
                                onClick={() => ask(s.text)}
                                className="group inline-flex items-center gap-2 rounded-lg border px-3 py-1.5 text-left text-sm transition-colors border-gray-200 bg-gray-50 text-gray-700 hover:bg-gray-100 hover:border-gray-300 dark:border-gray-700 dark:bg-gray-800/60 dark:text-gray-200 dark:hover:bg-gray-700/70 dark:hover:border-gray-600"
                            >
                                <span>{s.text}</span>
                                {s.hint && (
                                    <span className="text-[11px] text-gray-400 group-hover:text-gray-500 dark:text-gray-500 dark:group-hover:text-gray-400">
                                        {s.hint}
                                    </span>
                                )}
                            </button>
                        ))}
                    </div>
                    <p className="text-sm leading-relaxed text-gray-500 dark:text-gray-400">
                        It answers from{" "}
                        <a href={CORPUS_URL} target="_blank" rel="noopener noreferrer" className="underline decoration-gray-300 underline-offset-2 hover:decoration-gray-500 dark:decoration-gray-600 dark:hover:decoration-gray-400">
                            37 pages of the AI SDK docs
                        </a>{" "}
                        — Core, UI, Agents, Foundations, the v7 migration guide and the main reference pages —
                        and says so when a question falls outside them. Every answer shows the pages it
                        came from.{" "}
                        <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="underline decoration-gray-300 underline-offset-2 hover:decoration-gray-500 dark:decoration-gray-600 dark:hover:decoration-gray-400">
                            How it works, with measured results, on GitHub →
                        </a>
                    </p>
                </section>
            )}

            <div className="space-y-6">
                {messages.map((message) => {
                    const text = message.parts
                        .filter((p) => p.type === "text")
                        .map((p) => p.text)
                        .join("");

                    const sourcesPart = message.parts.find((p) => p.type === "data-sources");
                    const sources = sourcesPart?.type === "data-sources" ? sourcesPart.data : undefined;

                    // Sources stream BEFORE the model has answered, so the route cannot know
                    // whether it will refuse. Chunks can clear the 0.3 threshold and the prompt
                    // layer still (correctly) decline — as with "What is SDK?" at 0.35.
                    //
                    // Labelling those pills "Sources" would assert provenance for an answer that
                    // does not exist. But hiding them wastes real information: the retrieval was
                    // not useless, it just was not confident enough to answer from. So on a
                    // refusal they become "Related pages" — a recovery path instead of a dead end.
                    // This fixes the user's actual problem (being stuck) without touching
                    // retrieval, so it cannot weaken a guardrail.
                    const refused = isRefusal(text);

                    // "(Source 3)" in the answer → the pill whose chunks include 3. Pills are per
                    // page and chunks per section, so several numbers can point at one pill.
                    const pillId = (s: { id: number }) => `m-${message.id}-src-${s.id}`;
                    const citeTarget = (n: number) => {
                        const pill = sources?.find((s) => s.chunks.includes(n));
                        return pill ? pillId(pill) : null;
                    };

                    if (message.role === "user") {
                        return (
                            <div key={message.id} className="flex justify-end">
                                <div className="max-w-[80%] rounded-2xl rounded-br-md px-4 py-2.5 bg-blue-600 text-white shadow-sm">
                                    <p className="whitespace-pre-wrap leading-relaxed">{text}</p>
                                </div>
                            </div>
                        );
                    }

                    return (
                        <div key={message.id} className="flex justify-start">
                            <div className="max-w-[92%] rounded-2xl rounded-bl-md px-4 py-3 border shadow-sm border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900/70">
                                {/* Answer first, citations underneath — the reader wants the answer,
                                    then the provenance. Sources arrive on the stream before the text,
                                    so the order here is explicit rather than incidental. Block-level
                                    markdown (fences, lists, tables): components/markdown.tsx. */}
                                <Markdown text={text} citeTarget={citeTarget} />

                                {sources && sources.length > 0 && (
                                    <div className="mt-4 pt-3 border-t border-gray-200 dark:border-gray-800">
                                        <div className="mb-2 text-[11px] font-medium uppercase tracking-wide text-gray-400 dark:text-gray-500">
                                            {refused ? "Related pages" : "Sources"}
                                        </div>
                                        {refused && (
                                            <p className="mb-2.5 text-sm text-gray-500 dark:text-gray-400">
                                                I couldn&apos;t answer this from the documentation, but these pages
                                                might help.
                                            </p>
                                        )}
                                        <div className="flex flex-wrap gap-2">
                                            {sources.map((s) => (
                                                <a
                                                    key={s.id}
                                                    id={pillId(s)}
                                                    href={s.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    title={s.chunks.length > 1 ? `Sources ${s.chunks.join(", ")}` : `Source ${s.chunks[0]}`}
                                                    className="inline-flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg border transition-colors scroll-mt-24 border-gray-200 bg-gray-50 text-gray-700 hover:bg-gray-100 hover:border-gray-300 dark:border-gray-700 dark:bg-gray-800/60 dark:text-gray-200 dark:hover:bg-gray-700/70 dark:hover:border-gray-600"
                                                >
                                                    {!refused && (
                                                        <span className="tabular-nums text-[10px] text-gray-400 dark:text-gray-500">
                                                            {s.chunks.join(",")}
                                                        </span>
                                                    )}
                                                    <span>{s.title}</span>
                                                    {!refused && (
                                                        <span className="tabular-nums text-gray-400 dark:text-gray-500">
                                                            {s.score.toFixed(2)}
                                                        </span>
                                                    )}
                                                </a>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    );
                })}

                {/* Covers the silent gap while embed + rerank round-trip (~1s) */}
                {status === "submitted" && (
                    <div className="flex justify-start">
                        <div className="flex items-center gap-2 rounded-2xl rounded-bl-md px-4 py-3 border border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900/70">
                            <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-400 dark:bg-gray-500 animate-pulse" />
                            <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-400 dark:bg-gray-500 animate-pulse [animation-delay:150ms]" />
                            <span className="inline-block w-1.5 h-1.5 rounded-full bg-gray-400 dark:bg-gray-500 animate-pulse [animation-delay:300ms]" />
                            <span className="ml-1 text-sm text-gray-500 dark:text-gray-400">Searching the docs…</span>
                        </div>
                    </div>
                )}

                {error && rateLimit && (
                    // A rate limit is not an error the visitor caused — it's a budget
                    // decision. Amber and explanatory, not red and apologetic.
                    <div className="rounded-xl border p-4 text-sm border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-100">
                        <div className="font-medium">{rateLimit.title}</div>
                        <p className="mt-1 leading-relaxed text-amber-800 dark:text-amber-200/90">{rateLimit.body}</p>
                        <a
                            href={REPO_URL}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="mt-3 inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors border-amber-300 hover:bg-amber-100 dark:border-amber-800 dark:hover:bg-amber-900/40"
                        >
                            View the project on GitHub →
                        </a>
                    </div>
                )}

                {error && !rateLimit && (
                    <div className="rounded-xl border p-4 text-sm border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200">
                        <div className="font-medium">Something went wrong.</div>
                        <div className="mt-0.5 text-red-600 dark:text-red-300">The request failed. Please try again.</div>
                        <button
                            onClick={() => regenerate()}
                            className="mt-3 rounded-lg px-3 py-1.5 text-white transition-colors bg-red-600 hover:bg-red-700 dark:bg-red-700 dark:hover:bg-red-600"
                        >
                            Retry
                        </button>
                    </div>
                )}
            </div>

            <div ref={barRef} className="fixed inset-x-0 bottom-0 bg-gradient-to-t from-[var(--background)] via-[var(--background)] to-transparent pt-8 pb-6">
                <form
                    onSubmit={(e) => {
                        e.preventDefault();
                        ask(input);
                    }}
                    className="w-full max-w-3xl mx-auto px-4"
                >
                    <div className="relative">
                        <input
                            className="w-full px-4 py-3 pr-24 rounded-xl border shadow-sm outline-none transition-colors disabled:opacity-50 border-gray-300 bg-white text-gray-900 placeholder-gray-400 focus:border-blue-500 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100 dark:placeholder-gray-500 dark:focus:border-blue-500"
                            value={input}
                            placeholder={isBusy ? "Waiting for response…" : "Ask about the AI SDK…"}
                            onChange={(e) => setInput(e.target.value)}
                            disabled={isBusy}
                        />
                        {/* Stop aborts the stream client-side; useChat keeps whatever was already
                            streamed as the final message, so nothing is lost. */}
                        {isBusy && (
                            <button
                                type="button"
                                onClick={() => stop()}
                                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors border-gray-300 bg-white text-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200 dark:hover:bg-gray-700"
                            >
                                Stop
                            </button>
                        )}
                    </div>
                </form>
            </div>
        </div>
    );
}
