import { describe, expect, it } from "vitest";

import {
    answerHasScopeStatement,
    flatten,
    isQuoteFound,
    isScopeClaim,
    mustBeFactual,
    unlistedFactualSentences,
} from "@/evals/judge";

/**
 * Every case here is a shape that appeared in a real calibration run on Day 15, when the
 * judge went 12/12 false alarms → 0/12 and 0 → 2 → 1 → 0 missed lies over six runs. The
 * source below is a condensed stand-in for the real chunks those runs quoted.
 */
const SOURCE = `Otherwise, update both \`generateText\` and \`streamText\` calls:
\`\`\`tsx
const result = await generateText({ model, tools, activeTools: ['weather'], prompt: 'x' });
\`\`\`
generateText: Generates text for a given prompt and model. This function is ideal for non-interactive use cases such as automation tasks where you need to write text (e.g. drafting email or summarizing web pages) and for agents that use tools.
Let's start with a quick overview of the AI SDK, which is comprised of three parts:
- **AI SDK Core**: A unified, provider-agnostic API for generating text, structured objects, and tool calls with LLMs.
const { embeddings } = await embedMany({
  model: 'openai/text-embedding-3-small',
  values: ['a', 'b'],
});
The [streamText](/docs/x) function streams text generations from a language model. By default \`maxRetries\` is 2.`;
const HAY = flatten(SOURCE).toLowerCase();

describe("flatten", () => {
    it("strips markdown the judge cannot see: fences, links, inline delimiters", () => {
        expect(flatten("Use [`streamText`](/docs/x) to **stream**.")).toBe("Use streamText to stream.");
        // Fence markers go; the language tag stays — it is harmless in a haystack, and the
        // calibration was recorded against exactly this behaviour.
        expect(flatten("a\n\n```ts\nconst x = 1;\n```\n\nb")).toBe("a ts const x = 1; b");
    });
});

describe("isQuoteFound — exact and near-exact", () => {
    it("accepts a verbatim span and a punctuation-only difference", () => {
        expect(isQuoteFound("The streamText function streams text generations from a language model", HAY)).toBe(true);
        expect(isQuoteFound("AI SDK Core: A unified, provider agnostic API for generating text", HAY)).toBe(true); // hyphen normalised
    });

    it("accepts an explicitly elided quote and ignores sub-20-char fragments (`values: [...]`)", () => {
        expect(isQuoteFound("update both generateText and streamText calls: ... activeTools: ['weather'].", HAY)).toBe(true);
        expect(isQuoteFound("const { embeddings } = await embedMany({ model: 'openai/text-embedding-3-small', values: [...] });", HAY)).toBe(true);
    });

    it("accepts one clipped edge word, unless that word is an identifier", () => {
        expect(isQuoteFound("streamText function streams text generations from a language model.", HAY)).toBe(true);
        // 'streamText' swapped for generateText at the edge: the clipped word is an identifier
        expect(isQuoteFound("streamText is ideal for non-interactive use cases such as automation tasks", HAY)).toBe(false);
    });

    it("accepts one unmarked gap between two verbatim runs, in source order", () => {
        expect(
            isQuoteFound(
                "This function is ideal for non-interactive use cases such as automation tasks (e.g. drafting email or summarizing web pages) and for agents ",
                HAY
            )
        ).toBe(true);
    });

    it("rejects bag-of-words recombinations and every planted fabrication shape", () => {
        expect(isQuoteFound("streamText streams language model generations for chatbots and real-time applications", HAY)).toBe(false);
        expect(isQuoteFound("The generateText function streams text generations from a language model", HAY)).toBe(false);
        expect(isQuoteFound("For a non-streaming variant with the same options, call streamTextSync instead.", HAY)).toBe(false);
        expect(isQuoteFound("By default maxRetries is 5, so a failing request is retried five times before it throws.", HAY)).toBe(false);
        expect(isQuoteFound("This function is ideal for non-interactive use cases such as automation tasks and requires Node.js 24", HAY)).toBe(false);
    });

    it("rejects quotes too short to prove anything", () => {
        expect(isQuoteFound("streamText", HAY)).toBe(false);
        expect(isQuoteFound("", HAY)).toBe(false);
    });
});

describe("mustBeFactual — the guard on the model's claim labels", () => {
    it("forces factual for identifiers, numbers, braces and quoted values", () => {
        expect(mustBeFactual("Here's a basic example of using `streamText`:")).toBe(true);
        expect(mustBeFactual("For a non-streaming variant, call streamTextSync instead.")).toBe(true);
        expect(mustBeFactual("By default maxRetries is 5.")).toBe(true);
        expect(mustBeFactual("AI SDK 7 was released on 12 March 2026 and requires Node.js 24 or newer.")).toBe(true);
        expect(mustBeFactual("Pass { cache: 'aggressive' } in the settings.")).toBe(true);
        expect(mustBeFactual("Use the model 'openai/text-embedding-3-small' with embed.")).toBe(true);
    });

    it("does not force factual for scaffolding, apostrophes, or the bare product version", () => {
        expect(mustBeFactual("These updates enhance the SDK's functionality and streamline its usage.")).toBe(false);
        expect(mustBeFactual("In Vercel AI SDK 7, several changes and updates have been made.")).toBe(false);
        expect(mustBeFactual("The documentation doesn't cover error handling or specific model configurations.")).toBe(false);
    });
});

describe("scope statements", () => {
    it("answerHasScopeStatement — only the answer's own words allow a scope claim", () => {
        expect(answerHasScopeStatement("Use streamText. The documentation doesn't cover error handling.")).toBe(true);
        expect(answerHasScopeStatement("In AI SDK 7 several changes were made.")).toBe(false);
    });

    it("isScopeClaim — one clause, nothing asserted about the topic", () => {
        expect(isScopeClaim("The documentation doesn't cover additional configuration options for `streamText`.")).toBe(true);
        expect(isScopeClaim("The documentation doesn't cover X, but `fooSync` does it.")).toBe(false);
        expect(isScopeClaim("streamText streams text.")).toBe(false);
    });
});

describe("unlistedFactualSentences — the judge cannot skip a sentence", () => {
    const answer = `### How to Use \`streamText\`

You can use the \`streamText\` function to stream text. Here's a basic example:

\`\`\`ts
import { streamText } from 'ai';
const { textStream } = streamText({ model: __MODEL__, prompt: 'x' });
\`\`\`

- \`streamText\` is ideal for chatbots (Source 1).
- Use the \`onError\` callback to log errors.

The AI SDK is a powerful TypeScript library. Pass { cache: 'aggressive' } in the settings to have identical prompts served from cache. AI SDK 7 was released on 12 March 2026 and requires Node.js 24 or newer.`;
    const listed = [
        "You can use the streamText function to stream text generations from a language model.",
        "streamText is ideal for interactive use cases like chatbots.",
        "Use the onError callback to log errors.",
        "The AI SDK is a powerful TypeScript library.",
    ];

    it("flags exactly the planted sentences the judge left out — identifier-anchored and number-anchored", () => {
        expect(unlistedFactualSentences(answer, listed)).toEqual([
            "Pass { cache: 'aggressive' } in the settings to have identical prompts served from cache.",
            "AI SDK 7 was released on 12 March 2026 and requires Node.js 24 or newer.",
        ]);
    });

    it("is satisfied once a claim mentions the sentence's anchors", () => {
        expect(
            unlistedFactualSentences(answer, [
                ...listed,
                "Pass { cache: 'aggressive' } to enable caching.",
                "AI SDK 7 was released on 12 March 2026 and requires Node.js 24.",
            ])
        ).toEqual([]);
    });

    it("ignores code blocks — their lines are legitimately summarised into claims", () => {
        expect(unlistedFactualSentences("```ts\nconst x = streamText({ model: __MODEL__ });\n```", [])).toEqual([]);
    });
});
