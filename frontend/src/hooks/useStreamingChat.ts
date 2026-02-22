import { useState, useRef } from "react";
import type { Message } from "@/types";

export function useStreamingChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [conversationId, setConversationId] = useState<string | undefined>();
    const abortControllerRef = useRef<AbortController | null>(null);

    /** Parse buffered SSE lines and dispatch state updates.
     *  Returns any leftover partial data that hasn't formed a complete event yet. */
    const processSSEBuffer = (
        buffer: string,
        assistantMsgId: string,
        contentRef: { current: string }
    ): string => {
        const events = buffer.split("\n\n");
        // The last segment may be incomplete — keep it for the next read
        const remainder = events.pop() ?? "";

        for (const event of events) {
            const line = event.trim();
            if (!line.startsWith("data: ")) continue;
            const dataStr = line.slice(6);
            if (!dataStr) continue;

            try {
                const data = JSON.parse(dataStr);

                if (data.token) {
                    contentRef.current += data.token;
                    const snapshot = contentRef.current;
                    setMessages((prev) =>
                        prev.map((msg) =>
                            msg.id === assistantMsgId
                                ? { ...msg, content: snapshot }
                                : msg
                        )
                    );
                } else if (data.done) {
                    setMessages((prev) =>
                        prev.map((msg) =>
                            msg.id === assistantMsgId
                                ? {
                                    ...msg,
                                    isStreaming: false,
                                    metadata: data.metadata,
                                    sources: data.sources,
                                }
                                : msg
                        )
                    );
                } else if (data.error) {
                    setMessages((prev) =>
                        prev.map((msg) =>
                            msg.id === assistantMsgId
                                ? {
                                    ...msg,
                                    content: contentRef.current + `\n\n**Error:** ${data.error}`,
                                    isStreaming: false,
                                }
                                : msg
                        )
                    );
                }
            } catch {
                console.warn("Failed to parse SSE JSON chunk:", dataStr);
            }
        }

        return remainder;
    };

    /** Fallback: call the non-streaming /query endpoint and populate the message. */
    const fallbackToNonStreaming = async (
        question: string,
        currentConvId: string,
        assistantMsgId: string,
        signal: AbortSignal
    ) => {
        const response = await fetch("/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, conversation_id: currentConvId }),
            signal,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        setMessages((prev) =>
            prev.map((msg) =>
                msg.id === assistantMsgId
                    ? {
                        ...msg,
                        content: data.answer,
                        isStreaming: false,
                        metadata: data.metadata,
                        sources: data.sources,
                    }
                    : msg
            )
        );
    };

    const sendMessage = async (question: string) => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        abortControllerRef.current = new AbortController();

        const currentConvId = conversationId || crypto.randomUUID();
        setConversationId(currentConvId);

        const userMsgId = crypto.randomUUID();
        const assistantMsgId = crypto.randomUUID();

        setMessages((prev) => [
            ...prev,
            { id: userMsgId, role: "user", content: question },
            { id: assistantMsgId, role: "assistant", content: "", isStreaming: true },
        ]);
        setIsStreaming(true);

        try {
            // ── Primary: streaming SSE endpoint ──
            const response = await fetch("/query/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, conversation_id: currentConvId }),
                signal: abortControllerRef.current.signal,
            });

            if (!response.ok || !response.body) {
                throw new Error("stream-unavailable");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            const contentRef = { current: "" };
            let sseBuffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                sseBuffer += decoder.decode(value, { stream: true });
                sseBuffer = processSSEBuffer(sseBuffer, assistantMsgId, contentRef);
            }

            // Process any remaining buffered data after stream closes
            if (sseBuffer.trim()) {
                processSSEBuffer(sseBuffer + "\n\n", assistantMsgId, contentRef);
            }
        } catch (error: any) {
            if (error.name === "AbortError") return;

            // ── Fallback: non-streaming /query endpoint ──
            if (error.message === "stream-unavailable" || error.message?.includes("HTTP error")) {
                try {
                    console.info("Streaming unavailable, falling back to /query");
                    await fallbackToNonStreaming(
                        question,
                        currentConvId,
                        assistantMsgId,
                        abortControllerRef.current?.signal ?? new AbortController().signal
                    );
                    return;
                } catch (fallbackError: any) {
                    if (fallbackError.name === "AbortError") return;
                    // Fall through to connection error below
                }
            }

            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === assistantMsgId
                        ? {
                            ...msg,
                            content: msg.content + `\n\n**Connection Error:** Could not reach the server. Make sure the backend is running.`,
                            isStreaming: false,
                        }
                        : msg
                )
            );
        } finally {
            setIsStreaming(false);
            abortControllerRef.current = null;
        }
    };

    const stopStreaming = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
            setIsStreaming(false);
            setMessages((prev) => {
                const lastMsg = prev[prev.length - 1];
                if (lastMsg && lastMsg.isStreaming) {
                    return prev.map((msg, i) =>
                        i === prev.length - 1 ? { ...msg, isStreaming: false, content: msg.content + " [Stopped]" } : msg
                    );
                }
                return prev;
            });
        }
    }

    return { messages, isStreaming, sendMessage, stopStreaming };
}
