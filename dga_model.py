# dga_model.py
import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, cnn_out, kernel_size, lstm_hidden, lstm_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(embed_dim, cnn_out, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(cnn_out, lstm_hidden, lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.relu(self.conv1d(x)).permute(0, 2, 1)
        lstm_out, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out)
