use crate::audio::mel::MelSpec;

mod audio;

fn main() {
    let path = r"C:\Rust\shout\shout_train\data\sps-corpus-2.0-2025-12-05-de\audios\spontaneous-speech-de-71030.mp3";

    let pcm = audio::decoder::decode_to_f32_mono_16k(path).unwrap();

    let mel_spec:MelSpec = audio::mel::pcm_to_mel_frames_flat(&pcm, 80);

    println!(
        "mel: frames={}, mels={}, total={}",
        mel_spec.n_frames,
        mel_spec.n_mels,
        mel_spec.data.len()
    );

}
