"use client";

import { useState, useRef, useCallback } from "react";
import { ShaderAnimation } from "@/components/ui/shader-lines";
import { Starfield } from "@/components/ui/starfield";
import { Github, Mic, Send, Square } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  text: string;
}

export default function Home() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatOpen, setChatOpen] = useState(false);
  const [listening, setListening] = useState(false);
  const [intensity, setIntensity] = useState(1.0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const handleSend = useCallback(() => {
    const text = input.trim();
    if (!text) return;

    setMessages((prev) => [...prev, { role: "user", text }]);
    setInput("");

    // Pulse the sphere
    setIntensity(2.0);
    setTimeout(() => setIntensity(1.0), 600);

    // Simulated response
    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "I'm AEON — currently just a demo. Connect me to a server to start talking for real.",
        },
      ]);
    }, 1200);
  }, [input]);

  const toggleMic = () => {
    setListening((l) => !l);
    if (!listening) {
      setIntensity(1.8);
    } else {
      setIntensity(1.0);
    }
  };

  return (
    <main className="relative h-screen w-screen overflow-hidden bg-[#030712] flex flex-col items-center justify-center">
      {/* Starfield background */}
      <Starfield />

      {/* Aurora sphere — centered */}
      <div className="relative z-10 flex flex-col items-center gap-6">
        {/* Sphere container */}
        <div className="relative w-[340px] h-[340px] sm:w-[420px] sm:h-[420px]">
          <ShaderAnimation
            speed={0.05}
            intensity={intensity}
            className="w-full h-full absolute inset-0"
          />
          {/* Soft glow behind sphere */}
          <div className="absolute inset-0 rounded-full bg-sky-400/5 blur-3xl scale-125 pointer-events-none" />
        </div>

        {/* Branding */}
        <div className="text-center -mt-2">
          <h1 className="text-5xl sm:text-6xl font-display font-bold tracking-wider text-white/95">
            AEON
          </h1>
          <p className="mt-2 text-sm sm:text-base tracking-[0.25em] uppercase text-slate-400 font-light">
            The room that thinks
          </p>
        </div>

        {/* Chat area */}
        <div className="w-full max-w-md px-4 mt-4">
          {/* Messages */}
          {chatOpen && messages.length > 0 && (
            <div className="mb-3 max-h-48 overflow-y-auto space-y-2 scrollbar-thin scrollbar-thumb-slate-700">
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={`text-sm px-3 py-2 rounded-xl max-w-[85%] ${
                    m.role === "user"
                      ? "ml-auto bg-sky-500/20 text-sky-100"
                      : "mr-auto bg-slate-800/60 text-slate-200"
                  }`}
                >
                  {m.text}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}

          {/* Input bar */}
          <div className="flex items-center gap-2 bg-slate-900/60 backdrop-blur-md border border-slate-700/50 rounded-2xl px-4 py-2.5">
            <input
              type="text"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                if (!chatOpen) setChatOpen(true);
              }}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Talk to AEON..."
              className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-500 outline-none"
            />

            {/* Mic button */}
            <button
              onClick={toggleMic}
              className={`p-2 rounded-full transition-colors ${
                listening
                  ? "bg-red-500/20 text-red-400 animate-pulse"
                  : "hover:bg-slate-700/50 text-slate-400 hover:text-slate-200"
              }`}
              title={listening ? "Stop listening" : "Start listening"}
            >
              {listening ? <Square size={16} /> : <Mic size={16} />}
            </button>

            {/* Send button */}
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="p-2 rounded-full hover:bg-sky-500/20 text-slate-400 hover:text-sky-300 transition-colors disabled:opacity-30 disabled:pointer-events-none"
              title="Send"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* GitHub link — bottom right */}
      <a
        href="https://github.com/ansschh/cortex"
        target="_blank"
        rel="noopener noreferrer"
        className="fixed bottom-6 right-6 z-20 flex items-center gap-2 text-sm text-slate-500 hover:text-slate-300 transition-colors"
      >
        <Github size={18} />
        <span className="hidden sm:inline">GitHub</span>
      </a>

      {/* Bottom credit */}
      <p className="fixed bottom-6 left-1/2 -translate-x-1/2 z-20 text-[11px] text-slate-600 tracking-wide">
        Built at Caltech
      </p>
    </main>
  );
}
