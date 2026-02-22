import { useEffect, useRef } from "react";
import type { Message } from "@/types";
import { MessageBubble } from "./MessageBubble";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SmartSuggestions } from "./SmartSuggestions";

interface Props {
    messages: Message[];
    onSuggestionClick: (query: string) => void;
    isStreaming: boolean;
}

export function ChatArea({ messages, onSuggestionClick, isStreaming }: Props) {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isStreaming]);

    if (messages.length === 0) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-4 sm:p-6 w-full max-w-[100vw] overflow-x-hidden">
                <div className="max-w-2xl w-full text-center space-y-6 md:space-y-8">
                    <div className="relative p-6 px-4 sm:p-10 rounded-3xl bg-background/40 backdrop-blur-xl border border-white/20 dark:border-white/10 shadow-2xl overflow-hidden mb-6 md:mb-8 mx-auto w-full">
                        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-secondary/10 pointer-events-none" />
                        <div className="relative z-10 space-y-3">
                            <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight bg-gradient-to-br from-foreground to-foreground/70 bg-clip-text text-transparent px-2">
                                How can I help you today?
                            </h1>
                            <p className="text-muted-foreground text-base sm:text-lg max-w-lg mx-auto">
                                Ask anything about ClearPath's pricing, features, or documentation.
                            </p>
                        </div>
                    </div>
                    <SmartSuggestions onSelect={onSuggestionClick} />
                </div>
            </div>
        );
    }

    return (
        <ScrollArea className="flex-1 px-4 py-8">
            <div className="max-w-3xl mx-auto flex flex-col gap-2 pb-12">
                {messages.map((m) => (
                    <MessageBubble key={m.id} message={m} />
                ))}
                <div ref={bottomRef} />
            </div>
        </ScrollArea>
    );
}
