import glob
import os
import wave

clips_dir = "clips/wavs"

incorrect_rate = 0
incorrect_channels_value = 0
for wav_path in glob.glob(os.path.join(clips_dir, "*.wav")):
    with wave.open(wav_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        if sample_rate != 22050:
            incorrect_rate += 1
            print(f"{os.path.basename(wav_path)} HAD AN INCORRECT FRAMERATE: {sample_rate}")
        else:
            print(F"{os.path.basename(wav_path)}: {sample_rate}")
        data, sr = sf.read(wav_file)
        if len(data.shape) > 1 and data.shape[1] > 1:
            incorrect_channels_value += 1
            print(f"{fname}: {data.shape[1]} channels (stero/multi-channel)")
        else:
            print(f"{fname}: 1 channel (mono)")

if incorrect_rate == 0:
    print("All .wav files had a framerate of 22050 Hz.")

if incorrect_channels_value == 0:
    print("All .wav files had 1 channel (mono)")
        