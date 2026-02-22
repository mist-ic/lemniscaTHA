export interface Source {
    document: string;
    page?: number;
    relevance_score?: number;
    text?: string;
}

export interface MetadataTokens {
    input: number;
    output: number;
}

export interface Metadata {
    model_used: string;
    classification: "simple" | "complex";
    tokens: MetadataTokens;
    latency_ms: number;
    chunks_retrieved: number;
    evaluator_flags: string[];
}

export interface SearchResponse {
    answer: string;
    metadata: Metadata;
    sources: Source[];
    conversation_id?: string;
}

export interface Message {
    id: string;
    role: "user" | "assistant";
    content: string;
    isStreaming?: boolean;
    metadata?: Metadata;
    sources?: Source[];
}
