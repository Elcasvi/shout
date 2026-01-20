mod audio;

fn main() {
    let path = r"C:\Rust\shout\shout_train\data\sps-corpus-2.0-2025-12-05-de\audios\spontaneous-speech-de-71030.mp3";

    let pcm = audio::decoder::decode_to_f32_mono_16k(path).unwrap();

    let (mel, t) = audio::mel::pcm_to_mel_frames_flat(&pcm, 80);

    let min = mel.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let nan_count = mel.iter().filter(|x| x.is_nan()).count();

    println!("mel stats: min={min}, max={max}, nan_count={nan_count}");
    println!("frame 0: {:?}", &mel[0..80]);
    println!("frame 10: {:?}", &mel[10*80..11*80]);
}
