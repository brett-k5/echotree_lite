# Standard library imports
import sys, os
import csv

# Third party imports 
from TTS.tts.utils.text import cleaners

# Parameters
clips_dir = "clips/wavs"
output_csv = "metadata.csv"
cleaner_name = "english_cleaners"

# Get the cleaner function
text_cleaner = getattr(cleaners, cleaner_name)

# Prepare list of entries
metadata_entries = []

generator = os.walk(clips_dir)
root, dirs, files = next(generator) # There is only 1 directory so we need one set of (root, dirs, files)
for file in sorted(files):
    print(f"File: {file}")
    if file.endswith(".wav"):
        audio_path = os.path.join(root, file)
        # Corresponding transcript
        transcript_path = os.path.splitext(audio_path)[0] + ".txt"

        file = file.replace(".wav", "")

        if not os.path.exists(transcript_path):
            print(f"Missing transcript for {audio_path}, skipping")

        with open(transcript_path, "r", encoding="utf-8") as f:
            raw_text = f.read().strip()
        
        # Clean text
        cleaned_text = text_cleaner(raw_text)

        # Append to metadata.csv
        metadata_entries.append([file, raw_text, cleaned_text])
    
    elif file.endswith(".flac"):
        print("Stopping text cleaning. metadata.csv must include .wav file names only")
        break

for file in sorted(files):
    if file.endswith(".txt"):
        text_path = os.path.join(root, file)
        os.remove(text_path)
        print(f"{file} has been deleted.")

# Write to csv
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="|")
    for row in metadata_entries:
        writer.writerow(row)

print(f"metadata.csv created with {len(metadata_entries)} entries.")




