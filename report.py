import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dga_model import CNN_LSTM_Model
import json

MAX_SEQ_LEN = 100
EMBED_DIM = 128
CNN_OUT_CHANNELS = 64
KERNEL_SIZE = 3
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
vocab_size = len(vocab) + 1

model = CNN_LSTM_Model(vocab_size, EMBED_DIM, CNN_OUT_CHANNELS, KERNEL_SIZE,
                       LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, DROPOUT).to(device)
model.load_state_dict(torch.load("cnn_lstm_dga_model.pth", map_location=device))
model.eval()

import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DomainDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

data = pd.read_csv("D:/DGA_Detection/dga_domains_full.csv", header=None, names=['col1', 'col2', 'domain'])
data['domain'] = data['domain'].fillna('').astype(str)
data['label'] = data.apply(lambda row: 0 if row['col1'] == 'legit' or row['col2'] == 'alexa' else 1, axis=1)

def domain_to_seq(domain):
    seq = [vocab.get(ch, 0) for ch in domain]
    return seq[:MAX_SEQ_LEN] + [0] * max(0, MAX_SEQ_LEN - len(seq))

data['domain_seq'] = data['domain'].apply(domain_to_seq)
X = np.array(data['domain_seq'].tolist())
y = np.array(data['label'].tolist())
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
val_loader = DataLoader(DomainDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch).squeeze()
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).long()
        y_true.extend(y_batch.tolist())
        y_pred.extend(preds.cpu().tolist())

print("[Classification Report]\n")
print(classification_report(y_true, y_pred, target_names=["Legit", "DGA"]))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Legit", "DGA"], yticklabels=["Legit", "DGA"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
