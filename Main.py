import streamlit as st
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

st.set_page_config(page_title="News Similarity Model", layout="wide")

st.title("🧠 News Similarity Classification Dashboard")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ STATE ------------------
if "data" not in st.session_state:
    st.session_state.data = None

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "trained" not in st.session_state:
    st.session_state.trained = False

# ------------------ FUNCTIONS ------------------

def load_data():
    data = pd.read_csv("train.csv").dropna()
    data = data.sample(10000, random_state=42)
    data["text"] = data["title1_en"] + " [SEP] " + data["title2_en"]

    label_map = {"unrelated": 0, "agreed": 1}
    data = data[data["label"].isin(label_map)]
    data["label"] = data["label"].map(label_map)

    return data


@st.cache_resource
def load_model_components():
    tokenizer = BertTokenizerFast.from_pretrained("distilbert-base-uncased")
    bert_model = AutoModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, bert_model


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


# ------------------ SIDEBAR ------------------

st.sidebar.header("⚙️ Controls")

if st.sidebar.button("📂 Load Dataset"):
    with st.spinner("Loading dataset..."):
        st.session_state.data = load_data()
    st.sidebar.success("Dataset Loaded")

if st.sidebar.button("🤖 Load Model"):
    with st.spinner("Loading model..."):
        tokenizer, bert_model = load_model_components()
        st.session_state.tokenizer = tokenizer
        st.session_state.bert_model = bert_model
        st.session_state.model_loaded = True
    st.sidebar.success("Model Loaded")

# ------------------ TRAINING ------------------

if st.session_state.model_loaded and st.session_state.data is not None:

    if st.sidebar.button("🚀 Train Model"):

        tokenizer = st.session_state.tokenizer
        bert_model = st.session_state.bert_model
        data = st.session_state.data

        train_text, test_text, train_labels, test_labels = train_test_split(
            data["text"], data["label"], test_size=0.2, stratify=data["label"]
        )

        train_text, val_text, train_labels, val_labels = train_test_split(
            train_text, train_labels, test_size=0.1, stratify=train_labels
        )

        def encode(texts):
            return tokenizer(
                texts,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        train_tokens = encode(train_text.tolist())
        val_tokens = encode(val_text.tolist())
        test_tokens = encode(test_text.tolist())

        train_loader = DataLoader(TensorDataset(
            train_tokens["input_ids"],
            train_tokens["attention_mask"],
            torch.tensor(train_labels.values)
        ), batch_size=16, shuffle=True)

        val_loader = DataLoader(TensorDataset(
            val_tokens["input_ids"],
            val_tokens["attention_mask"],
            torch.tensor(val_labels.values)
        ), batch_size=16)

        test_loader = DataLoader(TensorDataset(
            test_tokens["input_ids"],
            test_tokens["attention_mask"],
            torch.tensor(test_labels.values)
        ), batch_size=16)

        model = NewsModel(bert_model).to(device)

        for p in model.bert.parameters():
            p.requires_grad = False

        for name, p in model.bert.named_parameters():
            if "transformer.layer.4" in name or "transformer.layer.5" in name:
                p.requires_grad = True

        weights = torch.tensor(1.0 / np.bincount(train_labels.values), dtype=torch.float).to(device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=2e-5
        )

        loss_fn = nn.NLLLoss(weight=weights)

        progress = st.progress(0)
        status = st.empty()

        train_losses = []
        val_accuracies = []

        best_acc = 0
        best_state = None

        for epoch in range(5):
            model.train()
            total_loss = 0

            status.text(f"Epoch {epoch+1}/5 Training...")

            for ids, mask, labels in train_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(ids, mask)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_losses.append(total_loss)

            model.eval()
            preds, actuals = [], []

            with torch.no_grad():
                for ids, mask, labels in val_loader:
                    ids, mask = ids.to(device), mask.to(device)
                    out = model(ids, mask)
                    p = torch.argmax(out, dim=1).cpu().numpy()
                    preds.extend(p)
                    actuals.extend(labels.numpy())

            acc = accuracy_score(actuals, preds)
            val_accuracies.append(acc)

            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()

            progress.progress((epoch + 1) / 5)

        model.load_state_dict(best_state)

        st.session_state.model = model
        st.session_state.test_loader = test_loader
        st.session_state.trained = True
        st.session_state.train_losses = train_losses
        st.session_state.val_accuracies = val_accuracies

        status.success("Training Complete")

# ------------------ RESULTS ------------------

if st.session_state.trained:

    st.subheader("📊 Training Curve")

    fig, ax = plt.subplots()
    ax.plot(st.session_state.train_losses, label="Loss")
    ax.plot(st.session_state.val_accuracies, label="Val Accuracy")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📈 Evaluation")

    model = st.session_state.model
    test_loader = st.session_state.test_loader

    preds, actuals = [], []

    with torch.no_grad():
        for ids, mask, labels in test_loader:
            ids, mask = ids.to(device), mask.to(device)
            out = model(ids, mask)
            p = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(p)
            actuals.extend(labels.numpy())

    st.text(classification_report(actuals, preds))

    cm = confusion_matrix(actuals, preds)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax2)
    st.pyplot(fig2)

    # ------------------ PREDICTION ------------------

    st.subheader("🔍 Try Prediction")

    text1 = st.text_input("First News")
    text2 = st.text_input("Second News")

    if st.button("Predict"):

        combined = text1 + " [SEP] " + text2

        tokenizer = st.session_state.tokenizer

        tokens = tokenizer(
            combined,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        ids = tokens["input_ids"].to(device)
        mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            out = model(ids, mask)
            probs = torch.exp(out).cpu().numpy()

        pred = np.argmax(probs)

        st.success(f"Prediction: {'TRUE' if pred == 1 else 'FALSE'}")
        st.write(f"Confidence: {probs[0][pred]:.4f}")

        st.subheader("🧪 LIME Explanation")

        explainer = LimeTextExplainer(class_names=["false", "true"])

        def predict_proba(texts):
            tokens = tokenizer(
                texts,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                out = model(tokens["input_ids"], tokens["attention_mask"])
                return torch.exp(out).numpy()

        exp = explainer.explain_instance(combined, predict_proba, num_features=8)

        words = [w for w, v in exp.as_list()][::-1]
        weights = [v for w, v in exp.as_list()][::-1]

        fig3, ax3 = plt.subplots()
        ax3.barh(words, weights)
        ax3.axvline(0)
        st.pyplot(fig3)