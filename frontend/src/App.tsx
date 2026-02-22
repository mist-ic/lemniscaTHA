import { useState, useEffect } from "react";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ChatArea } from "@/components/ChatArea";
import { InputArea } from "@/components/InputArea";
import { DebugPanel } from "@/components/DebugPanel";
import { ParticleBackground } from "@/components/ParticleBackground";
import { useStreamingChat } from "@/hooks/useStreamingChat";
import { TooltipProvider } from "@/components/ui/tooltip";

function App() {
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  const { messages, isStreaming, sendMessage, stopStreaming } = useStreamingChat();

  useEffect(() => {
    // Dark mode by default
    const savedTheme = localStorage.getItem("theme") as "light" | "dark" | null;
    const initialTheme = savedTheme || "dark";
    setTheme(initialTheme);
    document.documentElement.classList.toggle("dark", initialTheme === "dark");
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === "light" ? "dark" : "light";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
    document.documentElement.classList.toggle("dark", newTheme === "dark");
  };

  // Get the last assistant message that has metadata for the debug panel
  const lastAssistantMsg = [...messages].reverse().find(m => m.role === "assistant" && m.metadata);

  return (
    <TooltipProvider>
      <div className="flex flex-col h-screen bg-background text-foreground font-sans selection:bg-primary/20 relative overflow-hidden">

        {/* Ambient BioMed AI Glow Effects & Particles */}
        <div className="fixed inset-0 z-0 overflow-hidden bg-background">
          <ParticleBackground />
          <div className="absolute -top-[20%] -left-[10%] w-[50vw] h-[50vh] rounded-full bg-primary/10 blur-[120px] opacity-60 animate-pulse duration-10000 pointer-events-none" />
          <div className="absolute -bottom-[20%] -right-[10%] w-[60vw] h-[60vh] rounded-full bg-secondary/15 blur-[130px] opacity-50 pointer-events-none" />
        </div>

        {/* Header */}
        <header className="flex-none px-6 h-14 border-b bg-background/60 backdrop-blur-xl z-20 flex items-center justify-between sticky top-0 shadow-sm border-white/5 dark:border-white/10">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center border border-primary/20 shadow-sm">
              <span className="font-bold text-primary">CP</span>
            </div>
            <h1 className="font-semibold tracking-tight">ClearPath Support</h1>
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              className="rounded-full text-muted-foreground hover:text-foreground hover:bg-muted"
            >
              {theme === "dark" ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </Button>
            <DebugPanel lastMessage={lastAssistantMsg || null} />
          </div>
        </header>

        {/* Chat Area */}
        <main className="flex-1 overflow-hidden flex flex-col relative w-full pt-4">
          <ChatArea
            messages={messages}
            onSuggestionClick={sendMessage}
            isStreaming={isStreaming}
          />
        </main>

        {/* Input Area */}
        <div className="w-full relative z-10 pb-env-safe">
          <InputArea
            onSend={sendMessage}
            onStop={stopStreaming}
            disabled={isStreaming}
          />
        </div>
      </div>
    </TooltipProvider>
  );
}

export default App;
