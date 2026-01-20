use mel_spec::prelude::*;

/// Convert mono 16k PCM samples into a mel spectrogram matrix.
///
/// Returns an ndarray matrix with shape either (n_mels, frames) or (frames, n_mels)
/// depending on what `interleave_frames` returns in this crate.
pub fn pcm_to_mel_frames_flat(pcm_16k_mono: &[f32], n_mels: usize) -> (Vec<f32>, usize) {
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

    // IMPORTANT: in your version this returns Vec<f32> (as your error shows)
    let frames: Vec<f32> = interleave_frames(&mel_frames, false, 100);

    // Number of time frames T (assuming frames is laid out as T * n_mels)
    // (This is the layout expected by tga_8bit(frames, n_mels) in their example.)
    let n_frames = frames.len() / n_mels;

    (frames, n_frames)
}
