from pydub import AudioSegment
import csv
import glob
import json
import os

audio_dir = "greenlightsFlac/"
transcript_dir = "greenlightsFlac/"
output_root = "clips/"
metadata_file = os.path.join(output_root, "metadata.csv")
wave_file = "clips/wavs/"

os.makedirs(wave_file, exist_ok=True)

counter = 1

with open(metadata_file, "w", newline="", encoding="utf-8") as meta_csv:
    writer = csv.writer(meta_csv, delimiter = "|")

    for audio_path in glob.glob(os.path.join(audio_dir, "*.flac")):
        base = os.path.splitext(os.path.basename(audio_path))[0]
        transcript_path = os.path.join(transcript_dir, base + "_aligned.json")

        if not os.path.exists(transcript_path):
            print(f"there is no transcript file for {base}. Skipping to the next file")
            continue

        print(f"Processing {base}...")

        audio = AudioSegment.from_file(audio_path)

        with open(transcript_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        for seg in segments["segments"]:
            print(f"Segment_{counter}: {seg}")
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            clip = audio[start_ms: end_ms]
            clip = clip.set_frame_rate(22050).set_channels(1)

            clip_filename = os.path.join(wave_file, f"{counter:04d}.flac")
            clip.export(clip_filename.replace(".flac", ".wav"), format="wav")

            with open(clip_filename.replace(".flac", ".txt"), "w", encoding="utf-8") as tf:
                tf.write(seg["text"])
            
            writer.writerow([os.path.basename(clip_filename), seg["text"]])
            print(f"Saved {os.path.basename(clip_filename)}")
            counter += 1

print(f"\nDone! Exported {counter} clips to '{wave_file}'")
print(f"metadata.csv created at '{metadata_file}'")