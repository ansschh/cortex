const API_KEY = "sk_3d389eb62b782ad69b577f95f22acbaab332e5963a697a49";
const VOICE_ID = "KLON7Nwan8mJxpF2R8Yw";
const API_URL = `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}/stream`;

export interface AudioReactiveHandle {
  /** Current RMS amplitude 0..1 */
  getAmplitude: () => number;
  /** Stop playback */
  stop: () => void;
}

/**
 * Stream TTS from ElevenLabs → Web Audio API with an AnalyserNode
 * so we can read real amplitude for the aurora shader.
 */
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
      // Compute RMS
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

    // Collect the full audio buffer (stream comes as chunked mp3)
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
