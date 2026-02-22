import { Button } from "@/components/ui/button";

interface Props {
    onSelect: (query: string) => void;
}

const SUGGESTIONS = [
    "What are ClearPath's pricing plans?",
    "How do I set up Slack integration?",
    "What is ClearPath's PTO policy?",
    "How do I export my data?",
    "Tell me about keyboard shortcuts",
    "Compare Pro and Enterprise plans",
];

export function SmartSuggestions({ onSelect }: Props) {
    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 sm:gap-3 w-full px-2">
            {SUGGESTIONS.map((suggestion) => (
                <Button
                    key={suggestion}
                    variant="outline"
                    className="h-auto py-2.5 sm:py-3 px-3 sm:px-4 justify-start text-left font-normal bg-background/50 hover:bg-muted hover:text-foreground whitespace-normal break-words hover:border-primary/50 transition-colors text-sm"
                    onClick={() => onSelect(suggestion)}
                >
                    {suggestion}
                </Button>
            ))}
        </div>
    );
}
