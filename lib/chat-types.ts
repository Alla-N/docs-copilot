import { UIMessage, UIDataTypes } from "ai";

// Your custom data parts. The key becomes the part type: "data-sources".
export type ChatDataParts = {
    sources: {
        id: number;
        title: string;
        url: string;
        score: number;
    }[];
} & UIDataTypes;

export type ChatMessage = UIMessage<unknown, ChatDataParts>;
