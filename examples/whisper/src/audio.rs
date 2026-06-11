use rustfft::{FftPlanner, num_complex::Complex32};
use std::io::Cursor;
use std::path::Path;

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 80;
pub const N_SAMPLES: usize = 30 * SAMPLE_RATE; // 480_000
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000

/// Read a 16-bit / 32-bit / float WAV file, downmix to mono and resample to 16 kHz.
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    decode_wav(reader)
}

pub fn load_wav_bytes(bytes: &[u8]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::new(Cursor::new(bytes))?;
    decode_wav(reader)
}

fn decode_wav<R: std::io::Read>(
    mut reader: hound::WavReader<R>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    };

    // Downmix to mono
    let mono: Vec<f32> = if channels == 1 {
        samples
    } else {
        samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    // Resample to 16 kHz with simple linear interpolation if needed
    if spec.sample_rate as usize == SAMPLE_RATE {
        Ok(mono)
    } else {
        Ok(resample_linear(
            &mono,
            spec.sample_rate as usize,
            SAMPLE_RATE,
        ))
    }
}

fn resample_linear(input: &[f32], src_rate: usize, dst_rate: usize) -> Vec<f32> {
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((input.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let lo = pos.floor() as usize;
        let frac = (pos - lo as f64) as f32;
        let a = input[lo.min(input.len() - 1)];
        let b = input[(lo + 1).min(input.len() - 1)];
        out.push(a * (1.0 - frac) + b * frac);
    }
    out
}

pub fn pad_or_trim(audio: &[f32], length: usize) -> Vec<f32> {
    if audio.len() >= length {
        audio[..length].to_vec()
    } else {
        let mut out = audio.to_vec();
        out.resize(length, 0.0);
        out
    }
}

fn hz_to_mel_slaney(f: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4_f32.ln()) / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz_slaney(m: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0_f32;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4_f32.ln()) / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// Slaney-style mel filterbank that matches `librosa.filters.mel(sr, n_fft, n_mels)`.
/// Returned shape: (n_mels, n_fft/2 + 1).
pub fn mel_filters(sr: usize, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let n_freqs = n_fft / 2 + 1;
    let fmin = 0.0_f32;
    let fmax = sr as f32 / 2.0;

    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * (sr as f32 / 2.0) / (n_freqs as f32 - 1.0))
        .collect();

    let mel_min = hz_to_mel_slaney(fmin);
    let mel_max = hz_to_mel_slaney(fmax);
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| {
            let m = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
            mel_to_hz_slaney(m)
        })
        .collect();

    let fdiff: Vec<f32> = (0..n_mels + 1)
        .map(|i| mel_points[i + 1] - mel_points[i])
        .collect();

    let mut weights = vec![vec![0.0_f32; n_freqs]; n_mels];
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
        for j in 0..n_freqs {
            let lower = (fft_freqs[j] - mel_points[i]) / fdiff[i];
            let upper = (mel_points[i + 2] - fft_freqs[j]) / fdiff[i + 1];
            let v = lower.min(upper).max(0.0);
            weights[i][j] = v * enorm;
        }
    }
    weights
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

/// Compute log-mel spectrogram with whisper's preprocessing:
///   - reflect-pad input by N_FFT/2 on each side (matches torch.stft center=True)
///   - hann window of size N_FFT
///   - hop = HOP_LENGTH
///   - drop the last frame (matches whisper's stft[..., :-1])
///   - magnitudes squared, project through mel filterbank, log10
///   - clamp at max - 8.0 then (x + 4) / 4
///
/// Output shape: (n_mels, n_frames) flattened row-major.
pub fn log_mel_spectrogram(audio: &[f32], n_mels: usize) -> Vec<f32> {
    assert!(audio.len() == N_SAMPLES, "expected {} samples", N_SAMPLES);

    let pad = N_FFT / 2;
    let mut padded = vec![0.0_f32; audio.len() + 2 * pad];
    // Reflect padding (without endpoint repetition, matching torch.nn.functional.pad reflect)
    for i in 0..pad {
        padded[pad - 1 - i] = audio[i + 1];
    }
    padded[pad..pad + audio.len()].copy_from_slice(audio);
    let n = audio.len();
    for i in 0..pad {
        padded[pad + n + i] = audio[n - 2 - i];
    }

    let n_frames = (padded.len() - N_FFT) / HOP_LENGTH + 1;
    debug_assert!(n_frames > N_FRAMES);

    let window = hann_window(N_FFT);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let n_freqs = N_FFT / 2 + 1;

    // magnitudes^2: (n_freqs, n_frames - 1)  — drop the trailing frame at the end
    let used_frames = n_frames - 1;
    let mut magnitudes = vec![0.0_f32; n_freqs * used_frames];
    let mut buffer: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); N_FFT];

    for f in 0..used_frames {
        let start = f * HOP_LENGTH;
        for i in 0..N_FFT {
            buffer[i] = Complex32::new(padded[start + i] * window[i], 0.0);
        }
        fft.process(&mut buffer);
        for k in 0..n_freqs {
            let c = buffer[k];
            magnitudes[k * used_frames + f] = c.norm_sqr();
        }
    }

    // Apply mel filterbank: (n_mels, n_freqs) @ (n_freqs, used_frames) → (n_mels, used_frames)
    let filters = mel_filters(SAMPLE_RATE, N_FFT, n_mels);

    let mut log_spec = vec![0.0_f32; n_mels * used_frames];
    for m in 0..n_mels {
        for f in 0..used_frames {
            let mut acc = 0.0_f32;
            for k in 0..n_freqs {
                acc += filters[m][k] * magnitudes[k * used_frames + f];
            }
            log_spec[m * used_frames + f] = acc;
        }
    }

    // log10 with floor
    for v in log_spec.iter_mut() {
        *v = v.max(1e-10).log10();
    }
    // clamp at max - 8.0
    let max_val = log_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let floor_val = max_val - 8.0;
    for v in log_spec.iter_mut() {
        if *v < floor_val {
            *v = floor_val;
        }
    }
    // (x + 4) / 4
    for v in log_spec.iter_mut() {
        *v = (*v + 4.0) / 4.0;
    }

    log_spec
}
