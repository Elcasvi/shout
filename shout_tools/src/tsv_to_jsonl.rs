use anyhow::{Context, Result};
use serde::Serialize;
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};
#[derive(Debug, serde::Deserialize)]
struct Row {
    client_id: String,
    audio_file: String,
    duration_ms: String, 
    prompt_id: String,
    prompt: String,
    transcription: String,
    votes: String,
    age: String,
    gender: String,
    language: String,
    split: String,
    char_per_sec: String,
    quality_tags: String,
}

#[derive(Debug, Serialize)]
struct ManifestLine {
    audio_path: String,
    text: String,
    duration_ms: Option<u32>
}

pub fn convert()->Result<(), anyhow::Error>{
    println!("Converting TSV to JSONL");
    let dataset_root = PathBuf::from(r"C:\Rust\shout\shout_train\data\sps-corpus-2.0-2025-12-05-de");
    let tsv_path = dataset_root.join("ss-corpus-de.tsv");
    let audios_dir = dataset_root.join("audios");

    let out_path = PathBuf::from("manifests/train.jsonl");
    std::fs::create_dir_all(out_path.parent().unwrap())?;

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(&tsv_path)
        .with_context(|| format!("Failed to open TSV: {}", tsv_path.display()))?;

    let out_file = File::create(&out_path)
        .with_context(|| format!("Failed to create output: {}", out_path.display()))?;
    let mut writer = BufWriter::new(out_file);

    let mut kept = 0usize;
    let mut skipped_missing_audio = 0usize;
    let mut skipped_empty_prompt = 0usize;

    for result in rdr.deserialize::<Row>() {
        let row = result.context("Failed to parse a TSV row")?;

        let text = row.prompt.trim();
        if text.is_empty() {
            skipped_empty_prompt += 1;
            continue;
        }

        let audio_path = audios_dir.join(row.audio_file.trim());
        if !audio_path.exists() {
            skipped_missing_audio += 1;
            continue;
        }

        let duration_ms = row.duration_ms.trim().parse::<u32>().ok();

        let line = ManifestLine {
            audio_path: audio_path.to_string_lossy().to_string(),
            text: text.to_string(),
            duration_ms
        };

        serde_json::to_writer(&mut writer, &line)?;
        writer.write_all(b"\n")?;
        kept += 1;
    }

    writer.flush()?;

    println!("Wrote: {}", out_path.display());
    println!("Kept: {}", kept);
    println!("Skipped (empty prompt): {}", skipped_empty_prompt);
    println!("Skipped (missing audio file): {}", skipped_missing_audio);

    Ok(())
}

