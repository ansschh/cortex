"use client";

import { GlassEffect } from "./liquid-glass";
import { ExternalLink } from "lucide-react";

interface Project {
  name: string;
  repo: string;
  description: string;
  tags: string[];
}

const PROJECTS: Project[] = [
  {
    name: "Aeon",
    repo: "github.com/ansschh/cortex",
    description:
      "An AI-powered dorm room assistant that sees, hears, and responds. Whisper STT, Llama 3.3 reasoning, ElevenLabs TTS, MQTT/Home Assistant IoT, Spotify, email, calendar — all voice-controlled.",
    tags: ["Python", "FastAPI", "Whisper", "Llama", "MQTT", "ElevenLabs"],
  },
  {
    name: "RiskBench Suite",
    repo: "github.com/ansschh/riskbench-suite",
    description:
      "An end-to-end toolkit for benchmarking autonomous agents with a risk-first lens. Bundles task execution, risk metric computation, monitoring, a web dashboard, Dockerized runs, and Prometheus metrics.",
    tags: ["Python", "Docker", "Prometheus", "pip install"],
  },
  {
    name: "Netra",
    repo: "github.com/ansschh/Netra-Diabetic_Retinopathy_Detection-App",
    description:
      "A diabetic retinopathy screening app that analyzes fundus images to classify disease stage using deep learning. Uses a ResNet50-based pipeline with preprocessing and stage prediction for earlier detection and triage.",
    tags: ["Deep Learning", "ResNet50", "Medical Imaging"],
  },
  {
    name: "Dhanwantari",
    repo: "github.com/ansschh/Dhanwantari-Telemedicine_app",
    description:
      "An Android telemedicine app that streamlines rural healthcare with video consults, remote monitoring, and e-prescription workflows. Adds symptom-based ML triage, pulls real-time signals via Google Fit, and runs calls via Jitsi.",
    tags: ["Android", "Firebase", "Jitsi", "ML Triage"],
  },
  {
    name: "Mind It",
    repo: "github.com/ansschh/MindIt-Time_Management_App",
    description:
      "A voice-first time management app that turns speech into structured tasks and calendar events, then syncs them to Google Calendar. Combines Speech-to-Text with GPT-based parsing and Google Meet scheduling.",
    tags: ["STT", "GPT", "Google Calendar", "Firebase"],
  },
  {
    name: "Courtify",
    repo: "github.com/ansschh/Courtify-Facebook_For_Tennis_Lovers",
    description:
      "A tennis community app that shows real-time court availability and helps users find players by skill, age, and location. Estimates court occupancy using YOLOv8 plus OpenCV with live Firebase updates.",
    tags: ["YOLOv8", "OpenCV", "Firebase", "Android"],
  },
  {
    name: "Kanad",
    repo: "github.com/ansschh/Kanad-IoT_Enabled_Farming_App",
    description:
      "An IoT-enabled precision agriculture platform combining on-field sensors with ML to improve irrigation and fertilizer decisions. Forecasts soil moisture with LSTM, recommends crop actions from NPK readings, and runs ResNet50 crop disease detection.",
    tags: ["IoT", "Raspberry Pi", "LSTM", "Flask"],
  },
  {
    name: "Kepler",
    repo: "github.com/ansschh/kepler",
    description:
      "An Overleaf-style collaborative LaTeX editor built with Next.js, TailwindCSS, shadcn/ui, and Supabase. Delivers real-time co-editing with live PDF preview, templates, version history, and shareable links.",
    tags: ["Next.js", "Supabase", "LaTeX", "Real-time"],
  },
  {
    name: "Study Hub",
    repo: "github.com/ansschh/study-hub",
    description:
      "A collaborative study platform for organizing notes, resources, and study sessions with peers.",
    tags: ["Web App"],
  },
];

export function ProjectsPanel({
  onClose,
}: {
  onClose: () => void;
}) {
  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 pt-5 pb-4">
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-200 transition-colors text-sm"
          style={{ fontWeight: 300 }}
        >
          ← Back
        </button>
        <h2
          className="text-lg tracking-wider text-white/80"
          style={{ fontWeight: 300 }}
        >
          Projects
        </h2>
        <div className="w-12" />
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto px-6 pb-8 chat-scroll">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-4xl mx-auto">
          {PROJECTS.map((p, i) => (
            <a
              key={p.name}
              href={`https://${p.repo}`}
              target="_blank"
              rel="noopener noreferrer"
              className="block message-enter group"
              style={{ animationDelay: `${i * 0.06}s` }}
            >
              <GlassEffect className="rounded-2xl h-full hover:scale-[1.02] transition-transform duration-500">
                <div className="p-5 flex flex-col gap-3">
                  <div className="flex items-start justify-between">
                    <h3
                      className="text-base text-white/90"
                      style={{ fontWeight: 400 }}
                    >
                      {p.name}
                    </h3>
                    <ExternalLink
                      size={14}
                      className="text-slate-500 group-hover:text-sky-400 transition-colors mt-1 shrink-0"
                    />
                  </div>
                  <p
                    className="text-xs text-slate-400 leading-relaxed"
                    style={{ fontWeight: 300 }}
                  >
                    {p.description}
                  </p>
                  <div className="flex flex-wrap gap-1.5 mt-auto pt-1">
                    {p.tags.map((tag) => (
                      <span
                        key={tag}
                        className="text-[10px] px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-300/70 border border-sky-400/10"
                        style={{ fontWeight: 300 }}
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </GlassEffect>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
