import { Card } from "@/components/ui/card";
import type { Source } from "@/types";

interface Props {
    source: Source;
}

export function SourceCard({ source }: Props) {
    return (
        <Card className="p-3 text-xs bg-muted/50 hover:bg-muted transition-colors border-border/50">
            <div className="font-medium text-foreground mb-1 break-all">
                {source.document}
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
                {source.page && <span>Page {source.page}</span>}
                {source.relevance_score && (
                    <span>Score: {(source.relevance_score * 100).toFixed(0)}%</span>
                )}
            </div>
            {source.text && (
                <div className="mt-2 text-muted-foreground line-clamp-2 hover:line-clamp-none cursor-pointer">
                    {source.text}
                </div>
            )}
        </Card>
    );
}
