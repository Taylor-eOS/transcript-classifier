import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
import torch.nn.functional as F
from utils import TranscriptDataset, MAX_LENGTH

EPOCHS = 10
BATCH_SIZE = 32

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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    print("Model loaded.")
    optimizer = AdamW(model.parameters(), lr=4e-5)
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
        checkpoint_dir = f'checkpoints/epoch_{epoch+1}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        avg_loss = total_loss / len(train_loader)
        #print(f"  Average training loss: {avg_loss}")
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels_batch in val_loader:
                inputs = {key: val.to(device) for key, val in inputs.items()}
                labels_batch = labels_batch.to(device)
                outputs = model(**inputs, labels=labels_batch)
                val_loss = outputs.loss
                total_val_loss += val_loss.item()
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.size(0)
        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct / total
        print(f"  Validation loss: {avg_val_loss}")
        print(f"  Validation accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()


