/**
 * The corpus: which documentation pages docs-copilot knows. One list, imported by ingestion
 * (scripts/ingest.ts) and by the experiments that must run on the SAME pages
 * (scripts/experiments/chunking-comparison.ts) — a number measured on a different corpus than
 * the one deployed is not a number about this system.
 *
 * Adding a page here and running `npm run ingest -- --write` is the whole procedure.
 */
export const PAGES = [
    // Foundations — conceptual, high-value
    { title: "Foundations: Overview", url: "https://ai-sdk.dev/docs/foundations/overview.md" },
    { title: "Foundations: Providers and Models", url: "https://ai-sdk.dev/docs/foundations/providers-and-models.md" },
    { title: "Foundations: Prompts", url: "https://ai-sdk.dev/docs/foundations/prompts.md" },
    { title: "Foundations: Tools", url: "https://ai-sdk.dev/docs/foundations/tools.md" },
    { title: "Foundations: Streaming", url: "https://ai-sdk.dev/docs/foundations/streaming.md" },

    // Getting started
    { title: "Getting Started: Next.js App Router", url: "https://ai-sdk.dev/docs/getting-started/nextjs-app-router.md" },
    { title: "Navigating the Library", url: "https://ai-sdk.dev/docs/getting-started/navigating-the-library.md" },

    // Core — the meat
    { title: "Core: Overview", url: "https://ai-sdk.dev/docs/ai-sdk-core/overview.md" },
    { title: "Core: Generating Text", url: "https://ai-sdk.dev/docs/ai-sdk-core/generating-text.md" },
    { title: "Core: Generating Structured Data", url: "https://ai-sdk.dev/docs/ai-sdk-core/generating-structured-data.md" },
    { title: "Core: Tool Calling", url: "https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling.md" },
    { title: "Core: MCP Tools", url: "https://ai-sdk.dev/docs/ai-sdk-core/mcp-tools.md" },
    { title: "Core: Prompt Engineering", url: "https://ai-sdk.dev/docs/ai-sdk-core/prompt-engineering.md" },
    { title: "Core: Settings", url: "https://ai-sdk.dev/docs/ai-sdk-core/settings.md" },
    { title: "Core: Embeddings", url: "https://ai-sdk.dev/docs/ai-sdk-core/embeddings.md" },
    { title: "Core: Reranking", url: "https://ai-sdk.dev/docs/ai-sdk-core/reranking.md" },
    { title: "Core: Error Handling", url: "https://ai-sdk.dev/docs/ai-sdk-core/error-handling.md" },
    { title: "Core: Middleware", url: "https://ai-sdk.dev/docs/ai-sdk-core/middleware.md" },
    { title: "Core: Telemetry", url: "https://ai-sdk.dev/docs/ai-sdk-core/telemetry.md" },

    // Agents
    { title: "Agents: Overview", url: "https://ai-sdk.dev/docs/agents/overview.md" },
    { title: "Agents: Building Agents", url: "https://ai-sdk.dev/docs/agents/building-agents.md" },
    { title: "Agents: Workflow Patterns", url: "https://ai-sdk.dev/docs/agents/workflows.md" },
    { title: "Agents: Loop Control", url: "https://ai-sdk.dev/docs/agents/loop-control.md" },
    { title: "Agents: Memory", url: "https://ai-sdk.dev/docs/agents/memory.md" },

    // UI
    { title: "UI: Overview", url: "https://ai-sdk.dev/docs/ai-sdk-ui/overview.md" },
    { title: "UI: Chatbot", url: "https://ai-sdk.dev/docs/ai-sdk-ui/chatbot.md" },
    { title: "UI: Chatbot Tool Usage", url: "https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-tool-usage.md" },
    { title: "UI: Streaming Custom Data", url: "https://ai-sdk.dev/docs/ai-sdk-ui/streaming-data.md" },
    { title: "UI: Error Handling", url: "https://ai-sdk.dev/docs/ai-sdk-ui/error-handling.md" },

    // Reference — the functions you actually use
    { title: "Reference: streamText", url: "https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text.md" },
    { title: "Reference: generateText", url: "https://ai-sdk.dev/docs/reference/ai-sdk-core/generate-text.md" },
    { title: "Reference: embed", url: "https://ai-sdk.dev/docs/reference/ai-sdk-core/embed.md" },
    { title: "Reference: embedMany", url: "https://ai-sdk.dev/docs/reference/ai-sdk-core/embed-many.md" },
    { title: "Reference: tool", url: "https://ai-sdk.dev/docs/reference/ai-sdk-core/tool.md" },
    { title: "Reference: useChat", url: "https://ai-sdk.dev/docs/reference/ai-sdk-ui/use-chat.md" },
    { title: "Reference: convertToModelMessages", url: "https://ai-sdk.dev/docs/reference/ai-sdk-ui/convert-to-model-messages.md" },

    // Migration — explains the v7 changes you hit
    { title: "Migration: AI SDK 6 to 7", url: "https://ai-sdk.dev/docs/migration-guides/migration-guide-7-0.md" },
];
