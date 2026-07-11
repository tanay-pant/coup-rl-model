// frontend/src/audio.ts

let audioCtx: AudioContext | null = null;
let isMuted = false; // Default to unmuted until user interacts

export const initAudio = () => {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  }
  // Resume context if suspended (browser auto-play policy)
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
};

export const setMuted = (muted: boolean) => {
  isMuted = muted;
};

// Helper to play an oscillator sound
const playTone = (freq: number, type: OscillatorType, duration: number, vol: number = 0.1) => {
  if (isMuted || !audioCtx) return;
  try {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    
    // Quick fade out so it's not drawn out
    gain.gain.setValueAtTime(vol, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {
    // Ignore audio errors during rapid firing
  }
};

// Helper to play noise (for thuds, stabs, shuffles)
const playNoise = (duration: number, vol: number = 0.1, lowpassFreq: number = 1000) => {
  if (isMuted || !audioCtx) return;
  try {
    const bufferSize = audioCtx.sampleRate * duration;
    const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1;
    }
    
    const noiseSource = audioCtx.createBufferSource();
    noiseSource.buffer = buffer;
    
    const filter = audioCtx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = lowpassFreq;
    
    const gain = audioCtx.createGain();
    gain.gain.setValueAtTime(vol, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    
    noiseSource.connect(filter);
    filter.connect(gain);
    gain.connect(audioCtx.destination);
    
    noiseSource.start();
  } catch (e) {
    // Ignore audio errors
  }
};

export const Sounds = {
  hover: () => {
    // tiny click
    playTone(800, 'square', 0.05, 0.04);
  },
  duke: () => {
    // double beep, very short
    playTone(1200, 'square', 0.08, 0.15);
    setTimeout(() => playTone(1600, 'square', 0.1, 0.15), 60);
  },
  foreignAid: () => {
    // mid pitch
    playTone(1000, 'square', 0.08, 0.15);
    setTimeout(() => playTone(1400, 'square', 0.1, 0.15), 60);
  },
  income: () => {
    // lowest pitch
    playTone(800, 'square', 0.08, 0.15);
    setTimeout(() => playTone(1200, 'square', 0.1, 0.15), 60);
  },
  steal: () => {
    // police whistle / siren chirp
    if (isMuted || !audioCtx) return;
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(600, audioCtx.currentTime);
      osc.frequency.linearRampToValueAtTime(1200, audioCtx.currentTime + 0.1);
      osc.frequency.linearRampToValueAtTime(600, audioCtx.currentTime + 0.2);
      gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.2);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.2);
    } catch (e) {}
  },
  coup: () => {
    // sharp noise burst
    playNoise(0.1, 0.9, 2500);
  },
  assassinate: () => {
    // "snickt" sound matching the real audio profile: 
    // extremely fast (0.1s), high frequency dominant (4000Hz+), with reverb tail
    if (isMuted || !audioCtx) return;
    try {
      // 1. High frequency tone (the metallic ring)
      const osc = audioCtx.createOscillator();
      const oscGain = audioCtx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(4050, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(2000, audioCtx.currentTime + 0.1);
      
      oscGain.gain.setValueAtTime(1.35, audioCtx.currentTime);
      oscGain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.1);
      osc.connect(oscGain);
      oscGain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.1);

      // 2. High-pass noise burst (the scrape)
      const duration = 0.15;
      const bufferSize = audioCtx.sampleRate * duration;
      const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }
      const noise = audioCtx.createBufferSource();
      noise.buffer = buffer;
      
      const filter = audioCtx.createBiquadFilter();
      filter.type = 'highpass';
      filter.frequency.setValueAtTime(4000, audioCtx.currentTime); // High spectral centroid

      const noiseGain = audioCtx.createGain();
      noiseGain.gain.setValueAtTime(1.8, audioCtx.currentTime);
      noiseGain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);

      noise.connect(filter);
      filter.connect(noiseGain);
      noiseGain.connect(audioCtx.destination);
      noise.start();
    } catch (e) {}
  },
  shuffle: () => {
    // rapid noise bursts resembling cards swishing
    playNoise(0.06, 0.8, 2000);
    setTimeout(() => playNoise(0.05, 0.65, 1800), 80);
    setTimeout(() => playNoise(0.06, 0.65, 2000), 160);
  },
  thud: () => {
    // low deep retro drum / block sound
    if (isMuted || !audioCtx) return;
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'square'; // Square wave is much more audible on small speakers
      osc.frequency.setValueAtTime(200, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(50, audioCtx.currentTime + 0.15);
      gain.gain.setValueAtTime(0.72, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.15);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.15);
    } catch (e) {}
  },
  alert: () => {
    // challenge / reveal
    playTone(400, 'square', 0.06, 0.15);
    setTimeout(() => playTone(600, 'square', 0.08, 0.15), 60);
  },
  success: () => {
    // allow / accept
    playTone(600, 'sine', 0.04, 0.09);
    setTimeout(() => playTone(800, 'sine', 0.06, 0.09), 40);
  }
};
