import { Badge } from "@/components/ui/badge";

interface Props {
    flags: string[];
}

export function ConfidenceIndicator({ flags }: Props) {
    if (!flags || flags.length === 0) {
        return (
            <Badge variant="outline" className="bg-green-500/10 text-green-600 border-green-200 mt-2">
                <span>✅ High confidence</span>
            </Badge>
        );
    }

    return (
        <div className="flex flex-col gap-2 mt-2">
            {flags.includes("no_context") && (
                <Badge variant="outline" className="bg-amber-500/10 text-amber-600 border-amber-200 w-fit">
                    <span>⚠️ Low confidence — may not be based on documentation</span>
                </Badge>
            )}
            {flags.includes("refusal") && (
                <Badge variant="outline" className="bg-blue-500/10 text-blue-600 border-blue-200 w-fit">
                    <span>ℹ️ Information not found in documentation</span>
                </Badge>
            )}
            {flags.includes("conflicting_sources") && (
                <Badge variant="outline" className="bg-orange-500/10 text-orange-600 border-orange-200 w-fit">
                    <span>⚡ Conflicting information detected</span>
                </Badge>
            )}
            {flags.map((flag) => {
                if (!["no_context", "refusal", "conflicting_sources"].includes(flag)) {
                    return (
                        <Badge key={flag} variant="outline" className="bg-gray-500/10 text-gray-600 border-gray-200 w-fit">
                            <span>🚩 {flag}</span>
                        </Badge>
                    );
                }
                return null;
            })}
        </div>
    );
}
