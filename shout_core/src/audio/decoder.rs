use anyhow::{anyhow, Context, Result};
use std::path::Path;

use symphonia::core::{
    audio::{AudioBufferRef, SampleBuffer},
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    errors::Error as SymphoniaError,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

use rubato::{Fft, FixedSync, Resampler};
use audioadapter_buffers::direct::InterleavedSlice;

/// Decode an audio file to mono f32 samples at 16 kHz.
///
/// Returns: Vec<f32> where each element is one mono sample at 16_000 Hz.
pub fn decode_to_f32_mono_16k<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
    let path = path.as_ref();

    // -------------------------
    // 1) Decode with Symphonia
    // -------------------------
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open audio file: {}", path.display()))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Hint from extension (optional but helps).
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("unsupported format or failed to probe container")?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("no supported audio tracks found"))?;

    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("failed to create decoder for selected track")?;

    // We'll accumulate decoded interleaved f32 here.
    let mut interleaved_f32: Vec<f32> = Vec::new();

    // Determine input sample rate. Prefer codec params, but fall back to decoded buffer spec later.
    let mut input_sample_rate: Option<u32> = track.codec_params.sample_rate;

    // We also need to know channel count for downmixing.
    let mut input_channels: Option<usize> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::ResetRequired) => {
                return Err(anyhow!(
                    "decoder reset required (chained streams). handle by recreating decoder."
                ));
            }
            Err(SymphoniaError::IoError(_)) => break, // end of file
            Err(e) => return Err(e).context("error reading next packet"),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::IoError(_)) => continue,
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::ResetRequired) => {
                return Err(anyhow!(
                    "decoder reset required mid-stream. handle by recreating decoder."
                ));
            }
            Err(e) => return Err(e).context("unrecoverable decode error"),
        };

        // Update fallback info from decoded spec.
        input_sample_rate.get_or_insert(decoded.spec().rate);
        input_channels.get_or_insert(decoded.spec().channels.count());

        // Convert decoded buffer to interleaved f32
        let mut sbuf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        sbuf.copy_interleaved_ref(decoded);

        interleaved_f32.extend_from_slice(sbuf.samples());
    }

    let sr_in = input_sample_rate.ok_or_else(|| anyhow!("could not determine input sample rate"))?;
    let ch_in = input_channels.ok_or_else(|| anyhow!("could not determine channel count"))?;

    if interleaved_f32.is_empty() {
        return Err(anyhow!("decoded audio was empty"));
    }

    // -------------------------
    // 2) Downmix to mono
    // -------------------------
    let mono: Vec<f32> = if ch_in == 1 {
        interleaved_f32
    } else {
        let frames = interleaved_f32.len() / ch_in;
        let mut out = Vec::with_capacity(frames);

        for f in 0..frames {
            let mut sum = 0.0f32;
            let base = f * ch_in;
            for c in 0..ch_in {
                sum += interleaved_f32[base + c];
            }
            out.push(sum / ch_in as f32);
        }
        out
    };

    // -------------------------
    // 3) Resample to 16 kHz (if needed) using rubato v1.0.0
    // -------------------------
    const SR_OUT: usize = 16_000;

    if sr_in as usize == SR_OUT {
        return Ok(mono);
    }

    // Choose a chunk size for FFT resampler.
    // For offline processing, 1024 is a fine starting point.
    let chunk_size: usize = 1024;
    let sub_chunks: usize = 1;

    // Create FFT resampler (sync) for mono (1 channel).
    // rubato::Fft supports `process_all_into_buffer` which is perfect for full clips. :contentReference[oaicite:2]{index=2}
    let mut resampler = Fft::<f32>::new(
        sr_in as usize,
        SR_OUT,
        chunk_size,
        sub_chunks,
        1,                // mono
        FixedSync::Input, // fixed input chunking, output varies
    )
        .context("failed to construct FFT resampler")?;

    let input_len_frames = mono.len(); // mono => 1 sample per frame

    // Determine minimal output size (frames) needed. :contentReference[oaicite:3]{index=3}
    let out_len_frames = resampler.process_all_needed_output_len(input_len_frames);

    let mut out = vec![0.0f32; out_len_frames];

    // Adapters: (interleaved) with 1 channel => same as plain slice
    let input_adapter =
        InterleavedSlice::new(&mono, 1, input_len_frames).context("bad input adapter")?;

    let mut output_adapter =
        InterleavedSlice::new_mut(&mut out, 1, out_len_frames).context("bad output adapter")?;

    // Resample whole clip into preallocated buffer. :contentReference[oaicite:4]{index=4}
    let (_frames_read, frames_written) =
        resampler.process_all_into_buffer(&input_adapter, &mut output_adapter, input_len_frames, None)?;

    out.truncate(frames_written);
    Ok(out)
}
