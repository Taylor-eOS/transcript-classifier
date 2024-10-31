import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
import torch.nn.functional as F

EPOCHS = 10
MAX_LENGTH = 64

class TranscriptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Dataset initialized with {len(self.texts)} samples.")

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_length = torch.sum(encoding['input_ids'] != self.tokenizer.pad_token_id).item()
        #print(f"Token length for sample {idx}: {input_length}")
        return {key: val.squeeze() for key, val in encoding.items()}, torch.tensor(self.labels[idx], dtype=torch.long)

    def __len__(self):
        return len(self.texts)

def load_transcripts(file_path, label):
    transcripts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().strip('<>').strip()
            if line:
                transcripts.append(line)
    labels = [label] * len(transcripts)
    print(f"Loaded {len(transcripts)} transcripts from {file_path}.")
    return transcripts, labels

def main():
    file_A = 'transcriptions_A.txt'
    file_B = 'transcriptions_B.txt'
    transcripts_A, labels_A = load_transcripts(file_A, 0)
    transcripts_B, labels_B = load_transcripts(file_B, 1)
    texts = transcripts_A + transcripts_B
    labels = labels_A + labels_B
    print(f"Total transcripts: {len(texts)}")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    print("Tokenizer loaded.")
    dataset = TranscriptDataset(texts, labels, tokenizer, max_length=128)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    print("Model loaded.")
    optimizer = AdamW(model.parameters(), lr=5e-5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        total_loss = 0
        for batch_idx, (inputs, labels_batch) in enumerate(train_loader):
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}, Loss: {loss.item()}")
        avg_loss = total_loss / len(train_loader)
        print(f"  Average training loss: {avg_loss}")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs = {key: val.to(device) for key, val in inputs.items()}
                labels_batch = labels_batch.to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(F.softmax(logits, dim=1), dim=1)
                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.size(0)
        accuracy = correct / total
        print(f"  Validation Accuracy: {accuracy*100:.2f}%")
    output_dir = 'trained_distilbert'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}.")

if __name__ == "__main__":
    main()

