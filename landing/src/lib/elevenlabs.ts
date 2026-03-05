const API_KEY = process.env.NEXT_PUBLIC_ELEVENLABS_API_KEY ?? "";
const VOICE_ID = process.env.NEXT_PUBLIC_ELEVENLABS_VOICE_ID ?? "21m00Tcm4TlvDq8ikWAM";
const API_URL = `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}/stream`;

export interface AudioReactiveHandle {
  getAmplitude: () => number;
  stop: () => void;
}

export async function speakElevenLabs(
  text: string,
  onStart: () => void,
  onEnd: () => void
): Promise<AudioReactiveHandle> {
  const audioCtx = new AudioContext();
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.7;
  analyser.connect(audioCtx.destination);

  const dataArray = new Uint8Array(analyser.frequencyBinCount);

  const handle: AudioReactiveHandle = {
    getAmplitude: () => {
      analyser.getByteTimeDomainData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const v = (dataArray[i] - 128) / 128;
        sum += v * v;
      }
      return Math.sqrt(sum / dataArray.length);
    },
    stop: () => {
      audioCtx.close();
    },
  };

  if (!API_KEY) {
    console.warn("No ElevenLabs API key — falling back to browser TTS");
    // Fallback to browser TTS with simulated amplitude
    if (window.speechSynthesis) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.onstart = () => {
        (handle as any)._speaking = true;
        handle.getAmplitude = () => (handle as any)._speaking ? 0.1 + Math.random() * 0.2 : 0;
        onStart();
      };
      utterance.onend = () => { (handle as any)._speaking = false; onEnd(); };
      utterance.onerror = () => { (handle as any)._speaking = false; onEnd(); };
      handle.stop = () => { window.speechSynthesis.cancel(); audioCtx.close(); };
      window.speechSynthesis.speak(utterance);
    }
    return handle;
  }

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "xi-api-key": API_KEY,
      },
      body: JSON.stringify({
        text,
        model_id: "eleven_turbo_v2",
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
          style: 0.0,
          use_speaker_boost: true,
        },
      }),
    });

    if (!res.ok) {
      console.error("ElevenLabs error:", res.status, await res.text());
      onEnd();
      return handle;
    }

    const arrayBuffer = await res.arrayBuffer();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(analyser);

    source.onended = () => {
      onEnd();
      audioCtx.close();
    };

    onStart();
    source.start(0);
  } catch (err) {
    console.error("ElevenLabs TTS failed:", err);
    onEnd();
    audioCtx.close();
  }

  return handle;
}
