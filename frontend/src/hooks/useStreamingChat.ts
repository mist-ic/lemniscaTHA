import { useState, useRef } from "react";
import type { Message } from "@/types";

export function useStreamingChat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [conversationId, setConversationId] = useState<string | undefined>();
    const abortControllerRef = useRef<AbortController | null>(null);

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
            const response = await fetch("/query/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, conversation_id: currentConvId }),
                signal: abortControllerRef.current.signal,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            if (!response.body) {
                throw new Error("No response body");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantContent = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split("\n\n");

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const dataStr = line.slice(6);
                        if (!dataStr) continue;

                        try {
                            const data = JSON.parse(dataStr);

                            if (data.token) {
                                assistantContent += data.token;
                                setMessages((prev) =>
                                    prev.map((msg) =>
                                        msg.id === assistantMsgId
                                            ? { ...msg, content: assistantContent }
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
                                                content: assistantContent + `\n\n**Error:** ${data.error}`,
                                                isStreaming: false,
                                            }
                                            : msg
                                    )
                                );
                            }
                        } catch (e) {
                            console.warn("Failed to parse SSE JSON chunk:", dataStr);
                        }
                    }
                }
            }
        } catch (error: any) {
            if (error.name !== "AbortError") {
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
            }
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
