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

def predict_and_visualize(domain):
    seq = domain_to_seq(domain.lower(), vocab)
    input_tensor = torch.LongTensor([seq]).to(device)
    with torch.no_grad():
        output, attn_weights = model(input_tensor)
        prob = torch.sigmoid(output.squeeze()).item()

    chars = list(domain.lower())[:MAX_SEQ_LEN]
    attn_scores = attn_weights[0][:len(chars)].cpu().numpy()

    fig, ax = plt.subplots(figsize=(min(15, len(chars) * 0.6), 2.5))
    ax.bar(range(len(chars)), attn_scores, tick_label=chars, color='orange')
    ax.set_title(f"Attention Weights for \"{domain}\" (DGA Prob: {prob:.4f})")
    ax.set_xlabel("Characters"); ax.set_ylabel("Attention Score")
    plt.tight_layout()
    graph_path = f"report_{domain}.png"
    plt.savefig(graph_path); plt.close()

    return {
        "domain": domain,
        "probability": round(prob, 4),
        "risk_level": get_risk_level(prob),
        "graph_path": graph_path
    }

domains = [
    "google.com", "paypal.com", "xiuqweor.biz",
    "microsoft-auth.net", "lkjfdsaoiuqweop.biz", "zkxvqeorqwiep.biz"
]

results = []
for domain in domains:
    print(f"\n예측 중: {domain}")
    result = predict_and_visualize(domain)
    print(f"예측 확률: {result['probability']}, 위험 등급: {result['risk_level']}, 그래프 저장: {result['graph_path']}")
    results.append(result)

df = pd.DataFrame(results)
df.to_csv("dga_report.csv", index=False, encoding="utf-8-sig")

total = len(df)
high_risk = len(df[df['risk_level'].str.contains("매우 높음")])
mid_risk = len(df[df['risk_level'].str.contains("높음")])
low_risk = len(df[df['risk_level'].str.contains("낮음")])
normal = len(df[df['risk_level'].str.contains("보통")])

print("\n리포트 저장 완료: dga_report.csv")
print("\n모델 탐지 통계:")
print(f"총 테스트 수: {total}")
print(f"매우 위험 탐지: {high_risk}")
print(f"높음 탐지: {mid_risk}")
print(f"보통 탐지: {normal}")
print(f"낮음 탐지: {low_risk}")

df['predicted_label'] = (df['probability'] >= 0.5).astype(int)

dga_indicators = ["xiuqweor", "lkjfdsaoiuqweop", "zkxvqeorqwiep", "dhl-tracking", "microsoft-auth"]
legit_indicators = ["google", "paypal", "amazon"]

true_labels = []
for domain in df['domain']:
    if any(dga_key in domain for dga_key in dga_indicators):
        true_labels.append(1)  # DGA
    elif any(legit_key in domain for legit_key in legit_indicators):
        true_labels.append(0)  # Legit
    else:
        true_labels.append(0)  # 기본 Legit

df['true_label'] = true_labels

print("\n분류 평가 지표 (정밀도, 재현율, F1-score):")
report = classification_report(
    df['true_label'], df['predicted_label'], target_names=['Legit', 'DGA']
)
print(report)

print("""
[지표 설명]
- Precision (정밀도): DGA라고 예측한 것 중 실제 DGA인 비율
- Recall (재현율): 실제 DGA 중에서 모델이 맞게 DGA로 예측한 비율
- F1-score: Precision과 Recall의 조화 평균 (정확도+재현율 균형)
- Accuracy: 전체 중 맞게 예측한 비율
""")
