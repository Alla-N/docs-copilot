"use client";

import { useChat } from "@ai-sdk/react";
import { useState } from "react";

import { ChatMessage } from "@/lib/chat-types";

export default function Chat() {
    const [input, setInput] = useState("");
    const { messages, sendMessage, status, error, regenerate } = useChat<ChatMessage>();

    const isBusy = status === "submitted" || status === "streaming";

    return (
        <div className="flex flex-col w-full max-w-2xl py-12 mx-auto px-4">
            <div className="space-y-4 mb-32">
                {messages.map((message) => (
                    <div key={message.id}>
                        <div className="font-semibold">
                            {message.role === "user" ? "You" : "AI"}
                        </div>

                        {message.parts.map((part, i) => {
                            // Normal answer text
                            if (part.type === "text") {
                                return <div key={i}>{part.text}</div>;
                            }

                            // Structural marker we intentionally don't render
                            if (part.type === "step-start") {
                                return null;
                            }

                            // Custom sources data part (an array of sources)
                            if (part.type === "data-sources") {
                                // No cast: ChatMessage carries ChatDataParts, so narrowing
                                // to "data-sources" already gives part.data its real type.
                                const sources = part.data;
                                return (
                                    <div key={i} className="flex flex-wrap gap-2 mt-2">
                                        {sources.map((s) => (
                                            <a key={s.id} href={s.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1.5 px-3 py-1 text-xs rounded-full border transition-colors border-gray-300 bg-gray-50 text-gray-700 hover:bg-gray-100 hover:border-gray-400 dark:border-gray-700 dark:bg-gray-800/60 dark:text-gray-200 dark:hover:bg-gray-700/70 dark:hover:border-gray-500">
                                                📄 {s.title}
                                                <span className="text-gray-500 dark:text-gray-400 tabular-nums">{s.score.toFixed(2)}</span>
                                            </a>
                                        ))}
                                    </div>
                                );
                            }

                            return null;
                        })}
                    </div>
                ))}

                {/* Loader — shown during the silent pre-stream gap (embed + rerank round-trip) */}
                {status === "submitted" && (
                    <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400">
                        <span className="inline-block w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-pulse" />
                        <span className="inline-block w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-pulse [animation-delay:150ms]" />
                        <span className="inline-block w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-pulse [animation-delay:300ms]" />
                        <span className="ml-1 text-sm">Searching the docs…</span>
                    </div>
                )}

                {/* Error banner with retry */}
                {error && (
                    <div className="rounded border p-3 text-sm border-red-300 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200">
                        <div className="font-medium">Something went wrong.</div>
                        <div className="text-red-600 dark:text-red-300">The request failed. Please try again.</div>
                        <button onClick={() => regenerate()} className="mt-2 rounded px-3 py-1 text-white bg-red-600 hover:bg-red-700 dark:bg-red-700 dark:hover:bg-red-600">
                            Retry
                        </button>
                    </div>
                )}
            </div>

            <form
                onSubmit={(e) => {
                    e.preventDefault();
                    if (!input.trim() || isBusy) return;
                    sendMessage({ text: input });
                    setInput("");
                }}
                className="fixed bottom-0 w-full max-w-2xl mb-8 bg-[var(--background)] pt-2"
            >
                <input
                    className="w-full p-3 rounded border shadow outline-none transition-colors disabled:opacity-50 border-gray-300 bg-white text-gray-900 placeholder-gray-400 focus:border-gray-500 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100 dark:placeholder-gray-500 dark:focus:border-gray-500"
                    value={input}
                    placeholder={isBusy ? "Waiting for response…" : "Ask about the AI SDK…"}
                    onChange={(e) => setInput(e.target.value)}
                    disabled={isBusy}
                />
            </form>
        </div>
    );
}
