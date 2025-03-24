import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
from sklearn.metrics import classification_report

MAX_SEQ_LEN = 100
EMBED_DIM = 128
CNN_OUT_CHANNELS = 64
CNN_KERNEL_SIZE = 3
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 1
FC_HIDDEN_DIM = 64
DROPOUT = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
vocab_size = len(vocab) + 1

def domain_to_seq(domain, vocab):
    seq = [vocab.get(ch, 0) for ch in domain]
    return seq[:MAX_SEQ_LEN] + [0] * max(0, MAX_SEQ_LEN - len(seq))

class CNN_BiLSTM_Attn_Visual(nn.Module):
    def __init__(self, vocab_size, embed_dim, cnn_out, kernel_size, lstm_hidden, lstm_layers, fc_hidden, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(embed_dim, cnn_out, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_out, lstm_hidden, lstm_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_fc = nn.Linear(lstm_hidden * 2, 1, bias=False)
        self.fc1 = nn.Linear(lstm_hidden * 2, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def attention_weights(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.relu(self.conv1d(x)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)
        return attn_weights.squeeze(2), lstm_out

    def forward(self, x):
        attn_weights, lstm_out = self.attention_weights(x)
        context = torch.sum(attn_weights.unsqueeze(2) * lstm_out, dim=1)
        x = self.dropout(context)
        x = self.relu(self.fc1(x))
        return self.fc2(x), attn_weights

model = CNN_BiLSTM_Attn_Visual(
    vocab_size, EMBED_DIM, CNN_OUT_CHANNELS, CNN_KERNEL_SIZE,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, FC_HIDDEN_DIM, DROPOUT
).to(device)
model.load_state_dict(torch.load("cnn_bilstm_attn_dga_model.pth", map_location=device))
model.eval()

def get_risk_level(prob):
    if prob >= 0.9:
        return "매우 높음"
    elif prob >= 0.7:
        return "높음"
    elif prob >= 0.5:
        return "보통"
    else:
        return "낮음"

def predict_single_domain(domain):
    seq = domain_to_seq(domain.lower(), vocab)
    input_tensor = torch.LongTensor([seq]).to(device)
    with torch.no_grad():
        output, attn_weights = model(input_tensor)
        prob = torch.sigmoid(output.squeeze()).item()
    label = "DGA" if prob >= 0.5 else "Legit"
    risk = get_risk_level(prob)
    print(f"\n도메인: {domain}")
    print(f"예측 결과: {label} (확률: {prob:.4f})")
    print(f"위험 등급: {risk}")

if __name__ == "__main__":
    while True:
        user_input = input("\n분석할 도메인 입력 (종료하려면 q 입력): ").strip()
        if user_input.lower() == 'q':
            break
        predict_single_domain(user_input)
