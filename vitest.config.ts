import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

/**
 * Unit tests for the pure functions — the ones that carry this project's documented bugs.
 * Nothing here touches the network: modules that build a Supabase/OpenAI client at import
 * are given dummy env in tests/setup.ts so importing them is free.
 */
export default defineConfig({
    resolve: {
        alias: { "@": fileURLToPath(new URL("./", import.meta.url)) },
    },
    test: {
        include: ["tests/**/*.test.ts"],
        setupFiles: ["tests/setup.ts"],
        environment: "node",
    },
});
