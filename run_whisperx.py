import glob
import whisperx
from whisperx.utils import WriteJSON

# Path to flac files 
audio_files = sorted(glob.glob("GreenlightsFLAC/*.flac"))

model_name = "large"
device = "cpu"
compute_type = "float32"

model = whisperx.load_model(model_name, device=device, compute_type=compute_type)

for file in audio_files:
    print(f"Processing {file}...")
    result = model.transcribe(file)
    
    alignment_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(result["segments"], model=alignment_model, align_model_metadata=metadata, audio=file, device=device)

    # save aligned output
    output_path = file.replace(".flac", "_aligned.json")

    writer = WriteJSON(output_path)
    # Open the file where you want to save the JSON
    with open(output_path, "w", encoding="utf-8") as f:
        writer.write_result(result_aligned, f, options={})
