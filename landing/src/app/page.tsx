"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { ShaderAnimation } from "@/components/ui/shader-lines";
import { Starfield } from "@/components/ui/starfield";
import { GlassEffect, GlassFilter } from "@/components/ui/liquid-glass";
import { Github, Mic, Send, Square, ArrowLeft } from "lucide-react";
import { speakElevenLabs, type AudioReactiveHandle } from "@/lib/elevenlabs";
import { chat } from "@/lib/llm";

const EXPAND_THRESHOLD = 4;

interface Message {
  role: "user" | "assistant";
  text: string;
}

export default function Home() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatOpen, setChatOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [manualCollapse, setManualCollapse] = useState(false);
  const [listening, setListening] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [intensity, setIntensity] = useState(1.0);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const audioHandleRef = useRef<AudioReactiveHandle | null>(null);
  const animFrameRef = useRef<number | null>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (messages.length >= EXPAND_THRESHOLD && !expanded && !manualCollapse) {
      setExpanded(true);
    }
  }, [messages, expanded, manualCollapse]);

  /* ── Audio-reactive loop ─────────────────────────────────────── */

  const startAmplitudeLoop = useCallback(() => {
    if (animFrameRef.current) return;

    let smoothed = 1.0;

    const loop = () => {
      const handle = audioHandleRef.current;
      if (handle) {
        const amp = handle.getAmplitude();
        const target = 1.0 + amp * 12.0;
        smoothed += (target - smoothed) * 0.35;
      } else {
        smoothed += (1.0 - smoothed) * 0.08;
      }

      setIntensity(smoothed);

      if (audioHandleRef.current || Math.abs(smoothed - 1.0) > 0.02) {
        animFrameRef.current = requestAnimationFrame(loop);
      } else {
        animFrameRef.current = null;
        setIntensity(1.0);
      }
    };

    animFrameRef.current = requestAnimationFrame(loop);
  }, []);

  /* ── ElevenLabs TTS ──────────────────────────────────────────── */

  const speak = useCallback(
    async (text: string) => {
      audioHandleRef.current?.stop();
      audioHandleRef.current = null;

      const handle = await speakElevenLabs(
        text,
        () => {
          setSpeaking(true);
          startAmplitudeLoop();
        },
        () => {
          setSpeaking(false);
          audioHandleRef.current = null;
        }
      );
      audioHandleRef.current = handle;
    },
    [startAmplitudeLoop]
  );

  /* ── Send message → Llama → ElevenLabs ───────────────────────── */

  const handleSend = useCallback(
    async (text?: string) => {
      const msg = (text ?? input).trim();
      if (!msg) return;

      if (!chatOpen) setChatOpen(true);

      const newMessages: Message[] = [...messages, { role: "user", text: msg }];
      setMessages(newMessages);
      setInput("");

      setThinking(true);
      setIntensity(1.6);

      const response = await chat(newMessages);

      setThinking(false);
      setMessages((prev) => [...prev, { role: "assistant", text: response }]);
      speak(response);
    },
    [input, chatOpen, messages, speak]
  );

  /* ── Speech Recognition ──────────────────────────────────────── */

  const startListening = useCallback(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert("Speech recognition not supported — try Chrome.");
      return;
    }

    audioHandleRef.current?.stop();
    audioHandleRef.current = null;
    setSpeaking(false);

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event: any) => {
      let transcript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      setInput(transcript);

      const lastResult = event.results[event.results.length - 1];
      if (lastResult.isFinal) {
        setListening(false);
        recognitionRef.current = null;
        if (transcript.trim()) handleSend(transcript.trim());
      }
    };

    recognition.onerror = () => {
      setListening(false);
      recognitionRef.current = null;
    };
    recognition.onend = () => {
      setListening(false);
      recognitionRef.current = null;
    };

    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
    if (!chatOpen) setChatOpen(true);
    setIntensity(1.4);
  }, [chatOpen, handleSend]);

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    setListening(false);
    setIntensity(1.0);
  }, []);

  const toggleMic = useCallback(() => {
    if (listening) stopListening();
    else startListening();
  }, [listening, startListening, stopListening]);

  const goBack = useCallback(() => {
    setExpanded(false);
    setManualCollapse(true);
  }, []);

  /* ── Input bar ───────────────────────────────────────────────── */
  const inputBar = (
    <div className="flex items-center gap-2 bg-slate-900/50 backdrop-blur-md border border-slate-700/40 rounded-2xl px-4 py-2.5 netflix-transition">
      <input
        type="text"
        value={input}
        onChange={(e) => {
          setInput(e.target.value);
          if (!chatOpen) setChatOpen(true);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            handleSend();
          }
        }}
        placeholder="Talk to Aeon..."
        className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-500 outline-none"
        style={{ fontWeight: 300 }}
      />
      <button
        onClick={toggleMic}
        className={`p-2 rounded-full transition-colors ${
          listening
            ? "bg-red-500/20 text-red-400 animate-pulse"
            : "hover:bg-slate-700/50 text-slate-400 hover:text-slate-200"
        }`}
      >
        {listening ? <Square size={16} /> : <Mic size={16} />}
      </button>
      <button
        onClick={() => handleSend()}
        disabled={!input.trim() || thinking}
        className="p-2 rounded-full hover:bg-sky-500/20 text-slate-400 hover:text-sky-300 transition-colors disabled:opacity-30 disabled:pointer-events-none"
      >
        <Send size={16} />
      </button>
    </div>
  );

  const statusText = (listening || speaking || thinking) ? (
    <p className="text-center text-xs text-slate-500 mt-2 animate-pulse" style={{ fontWeight: 300 }}>
      {listening ? "Listening..." : thinking ? "Thinking..." : "Speaking..."}
    </p>
  ) : null;

  return (
    <main className="relative h-screen w-screen overflow-hidden bg-[#030712]">
      <GlassFilter />
      <Starfield />

      {/* ═══ AURORA ═══ */}
      <div
        className="absolute inset-0 z-[1] netflix-transition-slow"
        style={{ opacity: expanded ? 0.35 : 1 }}
      >
        <div
          className="absolute netflix-transition-slow"
          style={{
            top: expanded ? "-25vh" : "50%",
            left: expanded ? "-25vw" : "50%",
            width: expanded ? "150vw" : "420px",
            height: expanded ? "150vh" : "420px",
            transform: expanded ? "none" : "translate(-50%, calc(-50% - 80px))",
          }}
        >
          <ShaderAnimation
            speed={0.05}
            intensity={intensity}
            expanded={expanded}
            className="w-full h-full absolute inset-0"
          />
        </div>
        <div
          className="absolute pointer-events-none netflix-transition-slow"
          style={{
            top: "50%",
            left: "50%",
            width: expanded ? "100vw" : "500px",
            height: expanded ? "100vh" : "500px",
            transform: expanded
              ? "translate(-50%, -50%)"
              : `translate(-50%, calc(-50% - 80px)) scale(${1.15 + (intensity - 1) * 0.2})`,
            background: "radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%)",
            borderRadius: expanded ? "0" : "50%",
          }}
        />
      </div>

      {/* ═══ LANDING VIEW ═══ */}
      <div
        className="netflix-transition-slow absolute inset-0 z-10 flex flex-col items-center justify-center pointer-events-none"
        style={{
          opacity: expanded ? 0 : 1,
          transform: expanded ? "scale(0.9)" : "scale(1)",
          pointerEvents: expanded ? "none" : "auto",
        }}
      >
        <div className="w-[340px] h-[340px] sm:w-[420px] sm:h-[420px]" />

        <div className="text-center mt-6">
          <h1
            className="text-5xl sm:text-6xl font-display tracking-wider text-white/90"
            style={{ fontWeight: 300 }}
          >
            AEON
          </h1>
          <p
            className="mt-2 text-sm sm:text-base tracking-[0.25em] uppercase text-slate-400"
            style={{ fontWeight: 300 }}
          >
            The room that thinks
          </p>
        </div>

        <div className="w-full max-w-md px-4 mt-6">
          {chatOpen && messages.length > 0 && !expanded && (
            <div className="mb-3 max-h-48 overflow-y-auto space-y-3 chat-scroll">
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={`message-enter flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                  style={{ animationDelay: `${i * 0.05}s` }}
                >
                  {m.role === "assistant" ? (
                    <GlassEffect className="rounded-2xl max-w-[85%]">
                      <div className="px-3 py-2 text-sm text-slate-100" style={{ fontWeight: 300 }}>
                        {m.text}
                      </div>
                    </GlassEffect>
                  ) : (
                    <div
                      className="px-3 py-2 text-sm rounded-2xl max-w-[85%] bg-sky-500/15 text-sky-100 border border-sky-400/10"
                      style={{ fontWeight: 300 }}
                    >
                      {m.text}
                    </div>
                  )}
                </div>
              ))}
              <div ref={!expanded ? messagesEndRef : undefined} />
            </div>
          )}
          {inputBar}
          {statusText}
        </div>
      </div>

      {/* ═══ EXPANDED CHAT VIEW ═══ */}
      <div
        className="netflix-transition-slow absolute inset-0 z-20 flex flex-col"
        style={{
          opacity: expanded ? 1 : 0,
          transform: expanded ? "translateY(0)" : "translateY(40px)",
          pointerEvents: expanded ? "auto" : "none",
        }}
      >
        <div className="flex items-center justify-between px-6 pt-5 pb-3">
          <button
            onClick={goBack}
            className="flex items-center gap-2 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <ArrowLeft size={18} />
            <span className="text-sm" style={{ fontWeight: 300 }}>Back</span>
          </button>
          <h2 className="text-lg tracking-wider text-white/80" style={{ fontWeight: 300 }}>
            AEON
          </h2>
          <a
            href="https://github.com/ansschh/cortex"
            target="_blank"
            rel="noopener noreferrer"
            className="text-slate-500 hover:text-slate-300 transition-colors"
          >
            <Github size={18} />
          </a>
        </div>

        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto px-6 py-4 space-y-4 chat-scroll"
        >
          {messages.map((m, i) => (
            <div
              key={i}
              className={`message-enter flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              style={{ animationDelay: `${Math.min(i * 0.06, 0.3)}s` }}
            >
              {m.role === "assistant" ? (
                <GlassEffect className="rounded-2xl max-w-[80%] sm:max-w-[70%]">
                  <div className="px-4 py-3 text-sm text-slate-100" style={{ fontWeight: 300 }}>
                    {m.text}
                  </div>
                </GlassEffect>
              ) : (
                <div
                  className="px-4 py-3 text-sm rounded-2xl max-w-[80%] sm:max-w-[70%] bg-sky-500/15 text-sky-100 backdrop-blur-sm border border-sky-400/10"
                  style={{ fontWeight: 300 }}
                >
                  {m.text}
                </div>
              )}
            </div>
          ))}

          {thinking && (
            <div className="flex justify-start message-enter">
              <GlassEffect className="rounded-2xl">
                <div className="px-4 py-3 text-sm text-slate-400 flex items-center gap-2">
                  <span className="inline-flex gap-1">
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                    <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  </span>
                </div>
              </GlassEffect>
            </div>
          )}

          <div ref={expanded ? messagesEndRef : undefined} />
        </div>

        <div className="px-6 pb-6 pt-2">
          {inputBar}
          {statusText}
        </div>
      </div>

      {/* GitHub (landing only) */}
      <a
        href="https://github.com/ansschh/cortex"
        target="_blank"
        rel="noopener noreferrer"
        className="netflix-transition fixed bottom-6 right-6 z-30 flex items-center gap-2 text-sm text-slate-500 hover:text-slate-300 transition-colors"
        style={{
          opacity: expanded ? 0 : 1,
          pointerEvents: expanded ? "none" : "auto",
        }}
      >
        <Github size={18} />
        <span className="hidden sm:inline" style={{ fontWeight: 300 }}>GitHub</span>
      </a>
    </main>
  );
}
