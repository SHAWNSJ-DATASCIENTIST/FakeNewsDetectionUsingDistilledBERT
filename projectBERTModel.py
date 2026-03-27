import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv("train.csv").dropna()
data = data.sample(10000, random_state=42)

data["text"] = data["title1_en"] + " [SEP] " + data["title2_en"]

label_map = {"unrelated": 0, "agreed": 1}
data = data[data["label"].isin(label_map)]
data["label"] = data["label"].map(label_map)

train_text, test_text, train_labels, test_labels = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

train_text, val_text, train_labels, val_labels = train_test_split(
    train_text,
    train_labels,
    test_size=0.1,
    random_state=42,
    stratify=train_labels
)

tokenizer = BertTokenizerFast.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")

max_len = 64

def encode(texts):
    return tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

train_tokens = encode(train_text.tolist())
val_tokens = encode(val_text.tolist())
test_tokens = encode(test_text.tolist())

train_dataset = TensorDataset(
    train_tokens["input_ids"],
    train_tokens["attention_mask"],
    torch.tensor(train_labels.values)
)

val_dataset = TensorDataset(
    val_tokens["input_ids"],
    val_tokens["attention_mask"],
    torch.tensor(val_labels.values)
)

test_dataset = TensorDataset(
    test_tokens["input_ids"],
    test_tokens["attention_mask"],
    torch.tensor(test_labels.values)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

class NewsModel(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0]
        return torch.log_softmax(self.classifier(cls), dim=1)

model = NewsModel(bert_model).to(device)

for p in model.bert.parameters():
    p.requires_grad = False

for name, p in model.bert.named_parameters():
    if "transformer.layer.4" in name or "transformer.layer.5" in name:
        p.requires_grad = True

class_counts = np.bincount(train_labels.values)
weights = 1.0 / class_counts
weights = torch.tensor(weights, dtype=torch.float).to(device)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, weight_decay=0.01)
loss_fn = nn.NLLLoss(weight=weights)

train_losses = []
val_accuracies = []
best_val_acc = 0.0
best_state = None

for epoch in range(5):
    model.train()
    total_loss = 0

    for ids, mask, labels in train_loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(ids, mask)
        loss = loss_fn(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss)

    model.eval()
    val_preds = []
    val_actuals = []

    with torch.no_grad():
        for ids, mask, labels in val_loader:
            ids, mask = ids.to(device), mask.to(device)
            output = model(ids, mask)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_actuals.extend(labels.numpy())

    val_acc = accuracy_score(val_actuals, val_preds)
    val_accuracies.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("epoch", epoch + 1, "loss", total_loss, "val_acc", val_acc)

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for ids, mask, labels in test_loader:
        ids, mask = ids.to(device), mask.to(device)
        output = model(ids, mask)
        preds = torch.argmax(output, dim=1).cpu().numpy()
        predictions.extend(preds)
        actuals.extend(labels.numpy())

print(classification_report(actuals, predictions, target_names=["false", "true"]))

explainer = LimeTextExplainer(class_names=["false", "true"])

model_cpu = NewsModel(AutoModel.from_pretrained("distilbert-base-uncased"))
model_cpu.load_state_dict(model.state_dict())
model_cpu.to("cpu")
model_cpu.eval()

def predict_proba(texts):
    tokens = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        output = model_cpu(tokens["input_ids"], tokens["attention_mask"])
        probs = torch.exp(output)
    return probs.numpy()

plt.figure(figsize=(8, 5), dpi=1200)
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='train loss')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, marker='o', label='val accuracy')
plt.title("Training Curve")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.savefig("training_curve.png", dpi=1200, bbox_inches="tight")
plt.show()

cm = confusion_matrix(actuals, predictions)
plt.figure(figsize=(6, 5), dpi=600)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["false", "true"],
    yticklabels=["false", "true"]
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png", dpi=600, bbox_inches="tight")
plt.show()

plt.figure(figsize=(6, 4), dpi=1200)
sns.countplot(x=data["label"])
plt.title("Class Distribution")
plt.savefig("class_distribution.png", dpi=1200, bbox_inches="tight")
plt.show()

probs_list = []

with torch.no_grad():
    for ids, mask, labels in test_loader:
        ids, mask = ids.to(device), mask.to(device)
        output = model(ids, mask)
        probs = torch.exp(output).cpu().numpy()
        probs_list.extend(np.max(probs, axis=1))

plt.figure(figsize=(6, 4), dpi=300)
plt.hist(probs_list, bins=20)
plt.title("Prediction Confidence Distribution")
plt.savefig("confidence_hist.png", dpi=300, bbox_inches="tight")
plt.show()

while True:
    text1 = input("\nenter first news (or exit): ")
    if text1.lower() == "exit":
        break
    text2 = input("enter second news: ")

    combined = text1 + " [SEP] " + text2

    tokens = tokenizer(
        combined,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    ids = tokens["input_ids"].to(device)
    mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        output = model(ids, mask)
        probs = torch.exp(output).cpu().numpy()

    pred = np.argmax(probs)

    print("\nprediction:", "true" if pred == 1 else "false")
    print("confidence:", probs[0][pred])

    exp = explainer.explain_instance(combined, predict_proba, num_features=8, num_samples=50)

    feats = exp.as_list()
    words = [w for w, v in feats][::-1]
    weights = [v for w, v in feats][::-1]

    plt.figure(figsize=(8, 5), dpi=600)
    plt.barh(words, weights)
    plt.axvline(0)
    plt.title("LIME Explanation")
    plt.savefig("lime_explanation.png", dpi=600, bbox_inches="tight")
    plt.show()