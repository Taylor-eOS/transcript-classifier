import os
import whisper

ADD_FILENAME = False

def transcribe_file(model, file_path):
    try:
        result = model.transcribe(file_path)
        transcription = result['text'].replace('\n', ' ').strip()
        filename = os.path.basename(file_path)
        if ADD_FILENAME:
            return f"<{transcription}><{filename}>"
        else:
            return f"<{transcription}>"
    except Exception as e:
        return f"Error transcribing {file_path}: {e}"

def main(input_directory):
    #print(f'{input_directory.split("/")[1]}')
    output_file = f'transcriptions_{input_directory.split("/")[1]}.txt'
    if not os.path.isdir(input_directory):
        print(f"The directory '{input_directory}' does not exist.")
        return
    audio_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.wav')]
    total_files = len(audio_files)
    if total_files == 0:
        print("No .wav audio files found in the specified directory.")
        return
    print(f"Found {total_files} audio file(s) in '{input_directory}'. Loading Whisper model")
    model = whisper.load_model("tiny")
    print("Model loaded. Starting transcription")
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for idx, filename in enumerate(audio_files, 1):
            file_path = os.path.join(input_directory, filename)
            print(f"Processing {filename} ({idx}/{total_files})")
            transcription_line = transcribe_file(model, file_path)
            out_f.write(transcription_line + '\n')
    print(f"Transcription completed")

if __name__ == "__main__":
    main('train/A')
    main('train/B')

