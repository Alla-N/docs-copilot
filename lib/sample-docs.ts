export const sampleDocs = `# Streaming

streamText streams text generations from a language model. It returns tokens incrementally as they are generated, rather than waiting for the full response to complete. This is ideal for chat interfaces where you want to show output as it arrives.

You can consume the stream using the textStream property, which is an async iterable.

# Tool Calling

Tools let the model invoke functions you define. Each tool has a description, an input schema defined with Zod, and an execute function. The model decides when to call a tool based on the conversation.

When a tool is called, the SDK runs your execute function and feeds the result back to the model so it can continue generating.

# Embeddings

The embed function converts a single string into a vector of numbers. Use embedMany to embed multiple values in one call. Embeddings are used for semantic search and retrieval-augmented generation.

The cosineSimilarity function measures how similar two embedding vectors are, returning a value between -1 and 1.`;
