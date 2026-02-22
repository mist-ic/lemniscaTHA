import { Activity, Brain, CheckCircle2, ChevronRight, Clock, Cpu, FileText, Settings2, ShieldAlert, Zap } from "lucide-react";
import type { Message } from "@/types";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";

interface Props {
    lastMessage: Message | null;
}

export function DebugPanel({ lastMessage }: Props) {
    const meta = lastMessage?.metadata;

    if (!meta) {
        return (
            <Sheet>
                <SheetTrigger asChild>
                    <Button variant="outline" size="sm" className="gap-2 bg-background/50 backdrop-blur-sm border-primary/20 hover:bg-muted/80">
                        <Settings2 className="w-4 h-4 text-primary" />
                        <span className="hidden sm:inline">Debug Info</span>
                    </Button>
                </SheetTrigger>
                <SheetContent className="w-full sm:max-w-md bg-background/95 backdrop-blur-md border-l-primary/10">
                    <SheetHeader className="mb-6">
                        <SheetTitle className="flex items-center gap-2 text-primary">
                            <Activity className="w-5 h-5" />
                            Debug Telemetry
                        </SheetTitle>
                    </SheetHeader>
                    <div className="flex flex-col items-center justify-center p-8 text-center text-muted-foreground border rounded-xl bg-muted/30 border-dashed">
                        <Activity className="w-8 h-8 mb-3 opacity-20" />
                        <p>Waiting for first query to generate telemetry data...</p>
                    </div>
                </SheetContent>
            </Sheet>
        );
    }

    const is70B = meta.model_used.includes("70b");

    return (
        <Sheet>
            <SheetTrigger asChild>
                <Button variant="outline" size="sm" className="gap-2 bg-background/50 backdrop-blur-sm border-primary/20 hover:bg-muted/80">
                    <Settings2 className="w-4 h-4 text-primary" />
                    <span className="hidden sm:inline">Debug Info</span>
                </Button>
            </SheetTrigger>
            <SheetContent className="w-full sm:max-w-md bg-background/95 backdrop-blur-md border-l-primary/10 flex flex-col p-0">
                <SheetHeader className="p-6 pb-4 border-b">
                    <SheetTitle className="flex items-center gap-2 text-primary">
                        <Activity className="w-5 h-5" />
                        Debug Telemetry
                    </SheetTitle>
                </SheetHeader>

                <ScrollArea className="flex-1 p-6">
                    <div className="space-y-6 pb-6">
                        {/* Model Section */}
                        <div className="space-y-3">
                            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                                <Cpu className="w-4 h-4" /> Routing Decision
                            </h3>
                            <div className="p-4 rounded-xl bg-muted/50 border border-primary/10 space-y-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-sm font-medium">Classification</span>
                                    <div className="px-2.5 py-1 rounded-md bg-background border text-xs font-semibold capitalize flex items-center gap-1.5 shadow-sm">
                                        {meta.classification === 'complex' ? <Brain className="w-3.5 h-3.5 text-purple-500" /> : <Zap className="w-3.5 h-3.5 text-amber-500" />}
                                        {meta.classification}
                                    </div>
                                </div>
                                <div className="h-px bg-border/50 w-full" />
                                <div>
                                    <span className="text-xs text-muted-foreground block mb-1">Assigned Model</span>
                                    <div className="flex items-center gap-2 text-sm font-mono bg-background p-2 rounded-lg border shadow-inner">
                                        {is70B ? <Brain className="w-4 h-4 text-purple-500" /> : <Zap className="w-4 h-4 text-amber-500" />}
                                        {meta.model_used}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Performance Section */}
                        <div className="space-y-3">
                            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                                <Clock className="w-4 h-4" /> Performance Metrics
                            </h3>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="p-4 rounded-xl bg-muted/50 border border-primary/10 flex flex-col justify-between hidden">
                                    <span className="text-xs text-muted-foreground mb-1">Latency</span>
                                    <div className="flex items-end gap-1">
                                        <span className="text-2xl font-bold">{meta.latency_ms}</span>
                                        <span className="text-sm font-medium text-muted-foreground mb-0.5">ms</span>
                                    </div>
                                    <div className="w-full bg-background h-1.5 rounded-full mt-3 overflow-hidden border">
                                        <div
                                            className={`h-full rounded-full ${meta.latency_ms > 2000 ? 'bg-red-500' : meta.latency_ms > 1000 ? 'bg-amber-500' : 'bg-green-500'}`}
                                            style={{ width: `${Math.min(100, (meta.latency_ms / 3000) * 100)}%` }}
                                        />
                                    </div>
                                </div>

                                <div className="p-4 rounded-xl bg-muted/50 border border-primary/10 col-span-2">
                                    <div className="flex justify-between items-center mb-3">
                                        <span className="text-xs font-medium text-muted-foreground">Latency</span>
                                        <span className="text-sm font-bold bg-background px-2 py-0.5 rounded border shadow-sm">{meta.latency_ms} ms</span>
                                    </div>
                                    <div className="w-full bg-background h-1.5 rounded-full overflow-hidden border mb-4">
                                        <div
                                            className={`h-full rounded-full transition-all ${meta.latency_ms > 2000 ? 'bg-red-500' : meta.latency_ms > 1000 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                                            style={{ width: `${Math.min(100, (meta.latency_ms / 3000) * 100)}%` }}
                                        />
                                    </div>

                                    <div className="flex justify-between items-center mb-3 pt-3 border-t border-border/50">
                                        <span className="text-xs font-medium text-muted-foreground">Tokens</span>
                                        <span className="text-xs font-mono bg-background px-2 py-0.5 rounded border shadow-sm">
                                            {meta.tokens.input + meta.tokens.output} total
                                        </span>
                                    </div>

                                    <div className="flex items-center gap-1 w-full h-8 rounded-lg overflow-hidden border bg-background relative">
                                        <div
                                            className="h-full bg-blue-500/80 hover:bg-blue-500 transition-colors flex items-center justify-center text-[10px] font-bold text-white relative group"
                                            style={{ width: `${(meta.tokens.input / (meta.tokens.input + meta.tokens.output)) * 100}%` }}
                                        >
                                            {meta.tokens.input > 50 && `${meta.tokens.input} In`}
                                            <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity" />
                                        </div>
                                        <div
                                            className="h-full bg-indigo-500/80 hover:bg-indigo-500 transition-colors flex items-center justify-center text-[10px] font-bold text-white relative group"
                                            style={{ width: `${(meta.tokens.output / (meta.tokens.input + meta.tokens.output)) * 100}%` }}
                                        >
                                            {meta.tokens.output > 20 && `${meta.tokens.output} Out`}
                                            <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity" />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Content Pipeline */}
                        <div className="space-y-3">
                            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
                                <FileText className="w-4 h-4" /> Pipeline Stage
                            </h3>
                            <div className="p-4 rounded-xl bg-muted/50 border border-primary/10">
                                <div className="flex justify-between items-center mb-4">
                                    <span className="text-sm font-medium">Chunks Retrieved</span>
                                    <div className="w-8 h-8 rounded-full bg-background border flex items-center justify-center font-bold text-primary shadow-sm">
                                        {meta.chunks_retrieved}
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <span className="text-sm font-medium flex items-center justify-between">
                                        Evaluator Output
                                        {meta.evaluator_flags.length === 0 ? (
                                            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                                        ) : (
                                            <ShieldAlert className="w-4 h-4 text-amber-500" />
                                        )}
                                    </span>
                                    {meta.evaluator_flags.length === 0 ? (
                                        <div className="text-xs text-muted-foreground bg-background p-3 rounded-lg border shadow-sm flex items-start gap-2">
                                            <CheckCircle2 className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                                            All verification checks passed. Response is clean.
                                        </div>
                                    ) : (
                                        <div className="space-y-2">
                                            {meta.evaluator_flags.map(flag => (
                                                <div key={flag} className="text-xs font-mono bg-amber-500/10 text-amber-600 p-2.5 rounded-lg border border-amber-500/20 flex items-center gap-2">
                                                    <ChevronRight className="w-3 h-3 text-amber-600" />
                                                    {flag}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                    </div>
                </ScrollArea>
            </SheetContent>
        </Sheet>
    );
}
