import os
import shutil
import whisper
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pydub import AudioSegment
from utils import run_forgiving_scripts, TranscriptDataset, CLASSES, CHUNK_LENGTH, print_timestamps, concatenate_audio_chunks, convert_to_wav, split_audio

BATCH_SIZE = 32
EPOCHS = 3
ADD_FILENAME = False
PRINT_UNCORRECTED_PREDICTIONS = False
PRINT_CORRECTED_PREDICTIONS = False
KEEP_CHUNKS = False

def load_classification_model(model_dir='model'):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    return model, tokenizer, device

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
        print(f"Error transcribing {file_path}: {e}")
        return ""

def get_model(model_dir='model'):
    model, tokenizer, device = load_classification_model(model_dir)
    return model, tokenizer, device

def main_inference(input_audio_file):
    output_directory = os.path.splitext(os.path.basename(input_audio_file))[0]
    output_directory = os.path.join('chunks', output_directory)
    wav_file = convert_to_wav(input_audio_file)
    audio_chunks = split_audio(wav_file, output_folder=output_directory, chunk_length=CHUNK_LENGTH)
    whisper_model = whisper.load_model("tiny")
    classification_model, tokenizer, device = get_model('model')
    classification_results = []
    transcriptions = []
    print("Starting transcription of audio chunks...")
    for idx, chunk in enumerate(audio_chunks, 1):
        print(f"Transcribing chunk {idx}/{len(audio_chunks)}: {chunk}")
        transcription = transcribe_file(whisper_model, chunk)
        transcriptions.append(transcription)
    dataset = TranscriptDataset(transcriptions, [0] * len(transcriptions), tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    print("Starting classification of transcriptions...")
    classification_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            outputs = classification_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            pred_classes = predictions.cpu().tolist()
            classification_results.extend(pred_classes)
    if PRINT_UNCORRECTED_PREDICTIONS:
        print(f"Uncorrected predictions: {classification_results}")
    refined_results = run_forgiving_scripts(classification_results)
    if PRINT_CORRECTED_PREDICTIONS:
        print(f"Corrected predictions: {refined_results}")
    selected_chunks = [chunk for chunk, label in zip(audio_chunks, refined_results) if label == 'B']
    combined_audio = concatenate_audio_chunks(selected_chunks)
    output_wav = os.path.splitext(input_audio_file)[0] + '_cut.wav'
    combined_audio.export(output_wav, format='wav')
    output_mp3 = os.path.splitext(input_audio_file)[0] + '_cut.mp3'
    AudioSegment.from_wav(output_wav).export(output_mp3, format='mp3')
    print(f"Exported classified audio to {output_mp3}")
    total_length_ms = len(AudioSegment.from_wav(wav_file))
    print_timestamps(refined_results, CHUNK_LENGTH, total_length_ms)
    if not KEEP_CHUNKS:
        shutil.rmtree(output_directory)
        os.remove(wav_file)

if __name__ == "__main__":
    input_file = 'one.mp3'
    if not os.path.isfile(input_file):
        raise ValueError("File path is required for inference mode.")
    main_inference(input_file)

