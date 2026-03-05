const API_KEY = process.env.NEXT_PUBLIC_OPENROUTER_API_KEY ?? "";
const BASE_URL = "https://openrouter.ai/api/v1/chat/completions";
const MODEL = "meta-llama/llama-3.3-70b-instruct";

const SYSTEM_PROMPT = `You are Aeon, an AI-powered dorm room assistant. You're speaking on your landing page demo.

About you:
- You control smart home devices (MQTT, Home Assistant, Tasmota)
- You manage Spotify playback, emails (Gmail/Outlook), Google Calendar
- You set timers, alarms, reminders, and todos
- You help with studying (flashcards, quizzes)
- You check weather via OpenWeatherMap
- Your brain runs on Llama 3.3 via Groq, you hear with Whisper, you speak with ElevenLabs
- You're fully open source on GitHub (github.com/ansschh/cortex)
- You run on a laptop or Raspberry Pi with a mic array

Personality:
- Chill, friendly, concise — like talking to a smart roommate
- Keep responses to 1-3 sentences max since they'll be spoken aloud
- Don't use markdown, bullet points, or formatting — this is voice output
- Don't use emojis
- Be genuinely helpful and natural, not corporate`;

export async function chat(
  messages: { role: "user" | "assistant"; text: string }[]
): Promise<string> {
  if (!API_KEY) {
    return "No API key configured. Set NEXT_PUBLIC_OPENROUTER_API_KEY in your .env.local file.";
  }

  try {
    const res = await fetch(BASE_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${API_KEY}`,
        "HTTP-Referer": "https://aeon.vercel.app",
        "X-Title": "Aeon Landing Page",
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          ...messages.map((m) => ({ role: m.role, content: m.text })),
        ],
        max_tokens: 150,
        temperature: 0.7,
      }),
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      console.error("LLM error:", res.status, errText);
      return "Sorry, I couldn't process that right now. Try again in a sec.";
    }

    const data = await res.json();
    return (
      data.choices?.[0]?.message?.content?.trim() ??
      "Hmm, I didn't get a response. Try again."
    );
  } catch (err) {
    console.error("LLM fetch failed:", err);
    return "Something went wrong reaching my brain. Try again in a moment.";
  }
}
