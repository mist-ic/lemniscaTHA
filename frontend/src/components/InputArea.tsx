import { useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { Button } from "@/components/ui/button";
import { SendHorizontal, Square } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
    onSend: (message: string) => void;
    onStop?: () => void;
    disabled: boolean;
}

export function InputArea({ onSend, onStop, disabled }: Props) {
    const [input, setInput] = useState("");
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const handleSend = () => {
        if (input.trim() && !disabled) {
            onSend(input.trim());
            setInput("");
            // Reset height
            if (textareaRef.current) {
                textareaRef.current.style.height = "auto";
            }
        }
    };

    const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        setInput(e.target.value);
        e.target.style.height = "auto";
        e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
    };

    return (
        <div className="p-4 bg-background/80 backdrop-blur-sm border-t border-border">
            <div className="max-w-3xl mx-auto relative flex items-end gap-2 bg-muted/50 rounded-2xl border p-2 shadow-sm focus-within:ring-1 focus-within:ring-ring transition-shadow">
                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={handleInput}
                    onKeyDown={onKeyDown}
                    placeholder="Ask a question about ClearPath..."
                    className={cn(
                        "flex-1 max-h-[200px] min-h-[44px] bg-transparent resize-none outline-none py-3 px-3",
                        "text-foreground placeholder:text-muted-foreground",
                        "scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent"
                    )}
                    rows={1}
                    disabled={disabled}
                />
                {disabled && onStop ? (
                    <Button
                        onClick={onStop}
                        size="icon"
                        variant="destructive"
                        className="h-11 w-11 rounded-xl shrink-0 transition-all hover:scale-105 active:scale-95"
                        aria-label="Stop generating"
                    >
                        <Square className="h-4 w-4 fill-current" />
                    </Button>
                ) : (
                    <Button
                        onClick={handleSend}
                        disabled={!input.trim() || disabled}
                        size="icon"
                        className="h-11 w-11 rounded-xl shrink-0 transition-all hover:scale-105 active:scale-95"
                    >
                        <SendHorizontal className="h-5 w-5" />
                    </Button>
                )}
            </div>
        </div>
    );
}
