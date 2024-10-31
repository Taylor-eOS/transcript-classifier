import numpy as np
from transformers import DistilBertTokenizerFast
from train_model import MAX_LENGTH

def load_transcripts(file_path):
    transcripts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().strip('<>').strip()
            if line:
                transcripts.append(line)
    print(f"Loaded {len(transcripts)} transcripts from {file_path}.")
    return transcripts

def calculate_token_lengths(transcripts, tokenizer, max_length=MAX_LENGTH):
    token_lengths = []
    for text in transcripts:
        encoding = tokenizer(text, truncation=False, padding=False)
        token_length = len(encoding['input_ids'])
        token_lengths.append(token_length)

    # Calculate percentiles
    percentiles = np.percentile(token_lengths, np.arange(0, 101, 5))
    counts_per_percentile = [(np.sum((token_lengths >= percentiles[i]) & (token_lengths < percentiles[i + 1])))
                             for i in range(len(percentiles) - 1)]

    print("Number of files in each 5-percentile range:")
    for i in range(len(counts_per_percentile)):
        print(f"{i * 5}-{(i + 1) * 5}th percentile: {counts_per_percentile[i]} files")

    # Average padding for max_length
    padding_lengths = [max_length - length for length in token_lengths if length < max_length]
    avg_padding = sum(padding_lengths) / len(token_lengths) if padding_lengths else 0

    # Average truncation for max_length
    truncation_lengths = [length - max_length for length in token_lengths if length > max_length]
    avg_truncation = sum(truncation_lengths) / len(token_lengths) if truncation_lengths else 0

    print(f"Average padding (for max length {max_length}): {avg_padding}")
    print(f"Average truncation (for max length {max_length}): {avg_truncation}")

def main(filename):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    transcripts = load_transcripts(filename)
    print("Tokenizer loaded. Calculating token lengths...")
    calculate_token_lengths(transcripts, tokenizer)

if __name__ == "__main__":
    file_A = 'transcriptions_A.txt'
    file_B = 'transcriptions_B.txt'
    main(file_A)
    main(file_B)

