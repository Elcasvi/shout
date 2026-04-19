use mel_spec::prelude::*;

#[derive(Debug, Clone)]
pub struct MelSpec {
    /// Number of time frames (T)
    pub n_frames: usize,

    /// Number of mel bins (M = usually 80)
    pub n_mels: usize,

    /// Flat buffer in row-major order:
    /// data[frame * n_mels + mel_bin]
    pub data: Vec<f32>,
}


/// Convert mono 16k PCM samples into a mel spectrogram matrix.
///
/// Returns an ndarray matrix with shape either (n_mels, frames) or (frames, n_mels)
/// depending on what `interleave_frames` returns in this crate.
pub fn pcm_to_mel_frames_flat(pcm_16k_mono: &[f32], n_mels: usize) -> MelSpec{
    let fft_size = 400;
    let hop_size = 160;
    let sampling_rate = 16000.0;

    let mut stft = Spectrogram::new(fft_size, hop_size);
    let mut mel = MelSpectrogram::new(fft_size, sampling_rate, n_mels);

    // Same as docs example: Vec<Array2<f64>>
    let mut mel_frames: Vec<ndarray::Array2<f64>> = Vec::new();

    for chunk in pcm_16k_mono.chunks(hop_size) {
        // pad last hop
        let mut hop = vec![0.0f32; hop_size];
        hop[..chunk.len()].copy_from_slice(chunk);

        if let Some(fft_frame) = stft.add(&hop) {
            let mel_frame = mel.add(&fft_frame); // Array2<f64>
            mel_frames.push(mel_frame);
        }
    }

    // Flatten exactly like the docs example
    let flat: Vec<f32> = interleave_frames(&mel_frames, false, 100);

    let n_frames = flat.len() / n_mels;

    MelSpec {
        n_frames,
        n_mels,
        data: flat,
    }
}
