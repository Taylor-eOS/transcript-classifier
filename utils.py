import os
import torch
from torch.utils.data import Dataset
from pydub import AudioSegment

CLASSES = ['A', 'B']
CHUNK_LENGTH = 10240
MAX_LENGTH = 64

class TranscriptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt')
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def first_forgiving_heuristic(predictions):
    segments = []
    current_label = predictions[0]
    current_indices = [0]
    for i in range(1, len(predictions)):
        if predictions[i] == current_label:
            current_indices.append(i)
        else:
            segments.append((current_label, current_indices))
            current_label = predictions[i]
            current_indices = [i]
    segments.append((current_label, current_indices))
    def flip_outlier(segments, outlier_length):
        min_surround = outlier_length + 1
        for i in range(1, len(segments) - 1):
            label, indices = segments[i]
            prev_label, prev_indices = segments[i - 1]
            next_label, next_indices = segments[i + 1]
            if len(indices) == outlier_length and len(prev_indices) >= min_surround and len(next_indices) >= min_surround:
                new_label = 'B' if label == 'A' else 'A'
                if indices[0] + 1 == indices[-1] + 1:
                    print(f"Flipping chunk {indices[0] + 1} from {label} to {new_label}")
                else:
                    print(f"Flipping chunks {indices[0] + 1}-{indices[-1] + 1} from {label} to {new_label}")
                segments[i] = (new_label, indices)
        return segments
    for i in range(1, 6): #6 means up to 5
        segments = flip_outlier(segments, i)
    corrected_predictions = []
    for label, indices in segments:
        for _ in indices:
            corrected_predictions.append(label)
    return corrected_predictions

def second_forgiving_heuristic(predictions):
    def apply(predictions, target):
        opposite = 'A' if target == 'B' else 'B'
        for i in range(len(predictions) - 7 + 1):
            segment = predictions[i:i + 7]
            if segment[:2] == [target] * 2 and segment[2:5] == [opposite, target, opposite] and segment[5:] == [target] * 2:
                print(f"Found {target*2}{opposite}{target}{opposite}{target*2} pattern at indices {i + 1}-{i + 7}. Correcting to all {target}.")
                predictions[i + 2:i + 5] = [target] * 3
        return predictions
    predictions = apply(predictions, 'A')
    predictions = apply(predictions, 'B')
    return predictions

def third_forgiving_heuristic(predictions):
    def apply_length_10(predictions, target):
        opposite = 'A' if target == 'B' else 'B'
        i = 0
        while i <= len(predictions) - 10:
            segment = predictions[i:i + 10]
            if segment == [target]*3 + [opposite]*2 + [target] + [opposite] + [target]*3:
                print(f"Pattern {''.join([target]*3 + [opposite]*2 + [target] + [opposite] + [target]*3)} found at index {i + 3}")
                predictions[i:i + 10] = [target] * 10
            elif segment == [target]*3 + [opposite] + [target]*2 + [opposite] + [target]*3:
                print(f"Pattern {''.join([target]*3 + [opposite] + [target]*2 + [opposite] + [target]*3)} found at index {i + 3}")
                predictions[i:i + 10] = [target] * 10
            elif segment == [target]*3 + [opposite] + [target] + [opposite]*2 + [target]*3:
                print(f"Pattern {''.join([target]*3 + [opposite] + [target] + [opposite]*2 + [target]*3)} found at index {i + 3}")
                predictions[i:i + 10] = [target] * 10
            i += 1
        return predictions
    def apply_length_11(predictions, target):
        opposite = 'A' if target == 'B' else 'B'
        i = 0
        while i <= len(predictions) - 11:
            segment = predictions[i:i + 11]
            if segment == [target]*3 + [opposite] + [target]*2 + [opposite]*2 + [target]*3:
                print(f"Pattern {''.join([target]*3 + [opposite] + [target]*2 + [opposite]*2 + [target]*3)} found at index {i + 3}")
                predictions[i:i + 11] = [target] * 11
            elif segment == [target]*3 + [opposite]*2 + [target] + [opposite]*2 + [target]*3:
                print(f"Pattern {''.join([target]*3 + [opposite]*2 + [target] + [opposite]*2 + [target]*3)} found at index {i + 3}")
                predictions[i:i + 11] = [target] * 11
            elif segment == [target]*3 + [opposite]*2 + [target]*2 + [opposite] + [target]*3:
                print(f"Pattern {''.join([target]*3 + [opposite]*2 + [target]*2 + [opposite] + [target]*3)} found at index {i + 3}")
                predictions[i:i + 11] = [target] * 11
            i += 1
        return predictions
    def apply_length_12(predictions, target):
        opposite = 'A' if target == 'B' else 'B'
        i = 0
        while i <= len(predictions) - 12:
            segment = predictions[i:i + 12]
            if segment == [target]*3 + [opposite]*2 + [target]*2 + [opposite]*2 + [target]*3:
                print(f"Pattern {''.join([target]*3 + [opposite]*2 + [target]*2 + [opposite]*2 + [target]*3)} found at index {i + 3}")
                predictions[i:i + 12] = [target] * 12
            i += 1
        return predictions
    for target in CLASSES:
        predictions = apply_length_10(predictions, target)
        predictions = apply_length_11(predictions, target)
        predictions = apply_length_12(predictions, target)
    return predictions

def fourth_forgiving_heuristic(predictions):
    for i in range(len(predictions) - 10 + 1):  #10 is the length of AAAABABBBB
        segment = predictions[i:i + 10]
        if segment == ['A', 'A', 'A', 'A', 'B', 'A', 'B', 'B', 'B', 'B']:
            predictions[i:i + 10] = ['A'] * 4 + ['B'] * 6
            print(f"Flipped specific AAAABABBBB pattern.")
    return predictions

def run_forgiving_scripts(predictions):
    predictions = fourth_forgiving_heuristic(predictions)
    predictions = third_forgiving_heuristic(predictions)
    predictions = second_forgiving_heuristic(predictions)
    predictions = first_forgiving_heuristic(predictions)
    return predictions

def print_timestamps(corrected_predictions, chunk_length_ms, total_length_ms):
    boundary_points_sec = []
    for i in range(1, len(corrected_predictions)):
        if corrected_predictions[i] != corrected_predictions[i - 1]:
            boundary = i * chunk_length_ms / 1000
            boundary_points_sec.append(boundary)
    if boundary_points_sec and boundary_points_sec[0] == 0:
        boundary_points_sec = boundary_points_sec[1:]
    if boundary_points_sec and boundary_points_sec[-1] == total_length_ms / 1000:
        boundary_points_sec = boundary_points_sec[:-1]
    print(' '.join(map(lambda x: str(int(round(x))), boundary_points_sec)))

def split_audio(wav_file, output_folder, chunk_length=CHUNK_LENGTH):
    audio = AudioSegment.from_wav(wav_file)
    os.makedirs(output_folder, exist_ok=True)
    chunks = []
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i + chunk_length]
        chunk_filename = os.path.join(output_folder, f"chunk_{i}_{i + chunk_length}.wav")
        chunk.export(chunk_filename, format="wav")
        chunks.append(chunk_filename)
    return chunks

def convert_to_wav(input_file):
    audio = AudioSegment.from_file(input_file)
    wav_file = os.path.splitext(input_file)[0] + '.wav'
    audio.export(wav_file, format='wav')
    return wav_file

def concatenate_audio_chunks(chunks):
    combined = AudioSegment.empty()
    for chunk in chunks:
        combined += AudioSegment.from_wav(chunk)
    return combined

