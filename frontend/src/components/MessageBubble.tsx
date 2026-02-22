import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "@/types";
import { ConfidenceIndicator } from "./ConfidenceIndicator";
import { SourceCard } from "./SourceCard";
import { cn } from "@/lib/utils";

interface Props {
    message: Message;
}

export function MessageBubble({ message }: Props) {
    const isAssistant = message.role === "assistant";

    return (
        <div
            className={cn(
                "flex w-full mb-6",
                isAssistant ? "justify-start" : "justify-end"
            )}
        >
            <div
                className={cn(
                    "max-w-[85%] rounded-2xl px-5 py-4",
                    isAssistant
                        ? "bg-muted text-foreground corner-tl-none border shadow-sm"
                        : "bg-primary text-primary-foreground corner-tr-none px-5 py-3"
                )}
            >
                {isAssistant && message.isStreaming && !message.content ? (
                    <div className="flex gap-1 items-center h-6">
                        <span className="w-1.5 h-1.5 bg-foreground/50 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                        <span className="w-1.5 h-1.5 bg-foreground/50 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                        <span className="w-1.5 h-1.5 bg-foreground/50 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                    </div>
                ) : (
                    <div className={cn("prose prose-sm dark:prose-invert max-w-none break-words",
                        !isAssistant && "text-primary-foreground prose-p:text-primary-foreground prose-a:text-primary-foreground prose-strong:text-primary-foreground"
                    )}>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                        </ReactMarkdown>
                        {isAssistant && message.isStreaming && (
                            <span className="inline-block w-2 h-4 bg-foreground/50 animate-pulse ml-1 align-middle" />
                        )}
                    </div>
                )}

                {isAssistant && message.metadata && (
                    <div className="mt-4 pt-4 border-t border-border/50">
                        <ConfidenceIndicator flags={message.metadata.evaluator_flags} />
                    </div>
                )}

                {isAssistant && message.sources && message.sources.length > 0 && (
                    <div className="mt-4 flex flex-col gap-2">
                        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Sources</span>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                            {message.sources.map((source, i) => (
                                <SourceCard key={i} source={source} />
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
