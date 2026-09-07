"use client";

import { useMemo, useState, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { CITE_PREFIX, linkCitations } from "@/lib/citations";

/**
 * Renders an assistant answer as markdown.
 *
 * The first renderer was a 15-line regex for `code` and **bold** — "not a markdown library,
 * keeps the dependency list honest". That reasoning held until the model started emitting
 * fenced code blocks on nearly every "how do I" question, and a fence is BLOCK-level: the
 * regex swallowed the whole block into one inline span with a literal ```ts prefix. The
 * honest dependency is the parser, not the regex; react-markdown + remark-gfm are the standard
 * pair, and everything visible here is a component override, so the look stays ours.
 *
 * Why this works while streaming: react-markdown re-parses the full text on every chunk, and
 * CommonMark treats an unterminated fence as running to the end of the document — so a code
 * block renders as a code block from its first line, not as prose that later snaps into shape.
 *
 * Safety: react-markdown does not render raw HTML from the text (it is dropped), and unsafe
 * URL protocols are stripped by its default url transform. We go one step further on links:
 * only absolute http(s) hrefs become anchors. The docs chunks contain RELATIVE links
 * ("./generating-text") that the model copies verbatim; resolved against our origin they'd be
 * dead, so they render as plain text instead of a 404.
 */

/** Only absolute web links are clickable — a relative doc link has no valid target here. */
function safeHref(href: string | undefined): string | null {
    if (!href) return null;
    return /^https?:\/\//i.test(href) ? href : null;
}

function CodeBlock({ lang, code }: { lang?: string; code: string }) {
    const [copied, setCopied] = useState(false);

    async function copy() {
        try {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        } catch {
            /* clipboard unavailable (insecure context, permissions) — the button just does nothing */
        }
    }

    return (
        <div className="my-3 overflow-hidden rounded-lg border border-gray-200 bg-gray-50 dark:border-gray-700 dark:bg-gray-950/60">
            <div className="flex items-center justify-between border-b border-gray-200 px-3 py-1.5 dark:border-gray-800">
                <span className="text-[11px] font-medium uppercase tracking-wide text-gray-400 dark:text-gray-500">
                    {lang ?? "code"}
                </span>
                <button
                    type="button"
                    onClick={copy}
                    aria-label="Copy code"
                    className="rounded px-2 py-0.5 text-[11px] font-medium text-gray-500 transition-colors hover:bg-gray-200 hover:text-gray-800 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-100"
                >
                    {copied ? "Copied" : "Copy"}
                </button>
            </div>
            {/* overflow-x on the block, never on the page: a long line scrolls inside its box. */}
            <pre className="overflow-x-auto px-3 py-2.5 text-[13px] leading-relaxed">
                <code className="font-mono text-gray-800 dark:text-gray-200">{code}</code>
            </pre>
        </div>
    );
}

type Props = {
    text: string;
    /**
     * Maps a cited chunk number ("Source 3" → 3) to the DOM id of the pill that stands for
     * it, or null when there is none (sources not received yet, or a number the model made
     * up). Without a target the citation still renders as a superscript, just not as a link.
     */
    citeTarget?: (chunk: number) => string | null;
};

/**
 * Element overrides. On code: react-markdown ≥ 9 no longer passes `inline`; block code arrives
 * as <pre><code>, and the hast for a block always ends in "\n" while inline code never contains
 * one. `pre` is flattened away so `code` alone decides which of the two it is rendering.
 */
function buildComponents(citeTarget?: Props["citeTarget"]): Components {
    return {
        pre: ({ children }) => <>{children}</>,
        code: ({ className, children }) => {
            const text = String(children);
            const lang = /language-([\w-]+)/.exec(className ?? "")?.[1];
            const isBlock = Boolean(lang) || text.includes("\n");
            if (!isBlock) {
                return (
                    <code className="rounded px-1 py-0.5 text-[0.9em] font-mono bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200">
                        {text}
                    </code>
                );
            }
            return <CodeBlock lang={lang} code={text.replace(/\n$/, "")} />;
        },
        p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0">{children}</p>,
        ul: ({ children }) => <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>,
        ol: ({ children }) => <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>,
        li: ({ children }) => <li className="[&>p]:my-0">{children}</li>,
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        // The model occasionally emits headings; keep them modest — this is a chat bubble.
        h1: ({ children }) => <h3 className="mt-3 mb-1 font-semibold">{children}</h3>,
        h2: ({ children }) => <h3 className="mt-3 mb-1 font-semibold">{children}</h3>,
        h3: ({ children }) => <h4 className="mt-3 mb-1 font-semibold">{children}</h4>,
        a: ({ href, children }) => {
            if (href?.startsWith(CITE_PREFIX)) {
                const n = Number(href.slice(CITE_PREFIX.length));
                const target = citeTarget?.(n) ?? null;
                const cls =
                    "ml-0.5 inline-block rounded px-1 align-super text-[10px] font-medium leading-none tabular-nums " +
                    "bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300";
                if (!target) return <sup className={cls}>{n}</sup>;
                return (
                    <a
                        href={`#${target}`}
                        title={`Source ${n}`}
                        className={`${cls} no-underline hover:bg-blue-100 dark:hover:bg-blue-900/60`}
                        onClick={(e) => {
                            // Smooth-scroll to the pill without pushing a hash into the URL.
                            e.preventDefault();
                            document.getElementById(target)?.scrollIntoView({ behavior: "smooth", block: "center" });
                        }}
                    >
                        {n}
                    </a>
                );
            }
            const url = safeHref(href);
            if (!url) return <span>{children}</span>;
            return (
                <a
                    href={url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 underline decoration-blue-300 underline-offset-2 hover:decoration-blue-600 dark:text-blue-400 dark:decoration-blue-700 dark:hover:decoration-blue-400"
                >
                    {children}
                </a>
            );
        },
        blockquote: ({ children }) => (
            <blockquote className="my-2 border-l-2 border-gray-300 pl-3 text-gray-600 dark:border-gray-700 dark:text-gray-400">
                {children}
            </blockquote>
        ),
        // GFM tables (remark-gfm) — scroll inside their own box, same rule as code.
        table: ({ children }) => (
            <div className="my-3 overflow-x-auto">
                <table className="w-full border-collapse text-sm">{children}</table>
            </div>
        ),
        th: ({ children }) => (
            <th className="border-b border-gray-300 px-2 py-1 text-left font-semibold dark:border-gray-700">{children}</th>
        ),
        td: ({ children }) => <td className="border-b border-gray-200 px-2 py-1 align-top dark:border-gray-800">{children}</td>,
        hr: () => <hr className="my-3 border-gray-200 dark:border-gray-800" />,
    };
}

export function Markdown({ text, citeTarget }: Props): ReactNode {
    const components = useMemo(() => buildComponents(citeTarget), [citeTarget]);
    return (
        <div className="leading-relaxed text-[15px]">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
                {linkCitations(text)}
            </ReactMarkdown>
        </div>
    );
}
