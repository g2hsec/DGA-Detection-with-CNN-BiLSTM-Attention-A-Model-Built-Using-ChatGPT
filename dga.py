import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import json

MAX_SEQ_LEN = 100
EMBED_DIM = 128
CNN_OUT_CHANNELS = 64
CNN_KERNEL_SIZE = 3
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 1
FC_HIDDEN_DIM = 64
DROPOUT = 0.5
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

csv_path = r"D:\\DGA_Detection\\dga_domains_full.csv"
data = pd.read_csv(csv_path, header=None, names=['col1', 'col2', 'domain'])
data['domain'] = data['domain'].fillna('').astype(str)

def get_label(row):
    return 0 if row['col1'] == 'legit' or row['col2'] == 'al-exa' else 1
data['label'] = data.apply(get_label, axis=1)

print("Label 분포:")
print(data['label'].value_counts())

all_chars = set(''.join(data['domain'].tolist()))
vocab = {ch: idx + 1 for idx, ch in enumerate(sorted(all_chars))}
vocab_size = len(vocab) + 1

def domain_to_seq(domain):
    seq = [vocab.get(ch, 0) for ch in domain]
    return seq[:MAX_SEQ_LEN] + [0] * max(0, MAX_SEQ_LEN - len(seq))

data['domain_seq'] = data['domain'].apply(domain_to_seq)
X = np.array(data['domain_seq'].tolist())
y = np.array(data['label'].tolist())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

class DomainDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(DomainDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DomainDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

class CNN_BiLSTM_Attention_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, cnn_out, kernel_size,
                 lstm_hidden, lstm_layers, fc_hidden, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(embed_dim, cnn_out, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_out, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.attn_fc = nn.Linear(lstm_hidden * 2, 1, bias=False)

        self.fc1 = nn.Linear(lstm_hidden * 2, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def attention_net(self, lstm_output):
        attn_weights = self.attn_fc(lstm_output)       # (B, L, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.relu(self.conv1d(x)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context = self.attention_net(lstm_out)
        x = self.dropout(context)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN_BiLSTM_Attention_Model(vocab_size, EMBED_DIM, CNN_OUT_CHANNELS, CNN_KERNEL_SIZE,
                                   LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, FC_HIDDEN_DIM, DROPOUT).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_loss_hist, val_loss_hist, val_acc_hist, val_auc_hist = [], [], [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)
    train_loss_hist.append(train_loss)

    model.eval()
    val_loss, correct = 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()
            correct += (preds == y_batch.long()).sum().item()
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(y_batch.cpu().tolist())

    val_loss /= len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)
    val_auc = roc_auc_score(all_labels, all_probs)

    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)
    val_auc_hist.append(val_auc)

    print(f"[{epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_hist, label="Train Loss")
plt.plot(val_loss_hist, label="Val Loss")
plt.legend(); plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(val_acc_hist, label="Accuracy")
plt.plot(val_auc_hist, label="AUC")
plt.legend(); plt.title("Validation Accuracy & AUC")
plt.show()

torch.save(model.state_dict(), 'cnn_bilstm_attn_dga_model.pth')
with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab, f)

print("모델과 vocab 저장 완료.")
print("[최종 검증 리포트]")
print(classification_report(all_labels, [int(p >= 0.5) for p in all_probs], target_names=['Legit', 'DGA']))
