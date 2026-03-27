# 🚀 Fake News Detector: DistilBERT + LIME Explainer

**Classify news headline pairs as "Agreed" or "Unrelated" with 88% F1-score + visual explanations for EVERY prediction**

## 🎯 One-Liner Demo

"Trump wins election" + "Donald takes White House" → agreed (91% confidence) + LIME explanation chart

## 📋 Quick Start (30 seconds)

```bash
git clone https://github.com/your-username/FakeNews-DistilBERT.git
cd FakeNews-DistilBERT
pip install -r requirements.txt
python FakeNews-DistilBERT.py
```

| Feature         | This Project | BERT-base | Random |
| --------------- | ------------ | --------- | ------ |
| F1-Score        | 88%          | 88%       | 50%    |
| Inference Speed | 26ms         | 42ms      | 😴     |
| Parameters      | 66M          | 110M      | 🤷     |
| Explanations    | LIME charts  | ❌        | ❌     |
| Interactive     | ✅ Live demo | ❌        | ❌     |

🧠 Architecture
[Headline1] [SEP] [Headline2] → DistilBERT → [CLS] → Classifier → "agreed" + LIME viz

               precision    recall  f1-score   support
     Unrelated       0.91      0.88      0.89      1124
       Agreed       0.84      0.87      0.86       876
    accuracy                           0.88      2000

DistilBERT = BERT performance, 40% lighter, 46% faster.

🎮 Live Demo

Enter headline 1: COVID vaccine proven safe
Enter headline 2: Scientists confirm vaccine safety

✅ agreed (0.91 confidence) + [LIME chart]
"vaccine" █████████ +0.38 | "safe" ████████ +0.32

🗂️ Files
text
FakeNews-DistilBERT/
├── FakeNews-DistilBERT.py
├── train.csv
├── test.csv
└── requirements.txt

⚙️ Install
bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
🧑‍💻 Specs
text
distilbert-base-uncased | 6 layers | 12 heads | 768 dim | 128 tokens
Dropout(0.3) → 768→256→2 | AdamW 1e-5 | 2 epochs | batch 16
🔍 LIME Example
text
Prediction: agreed (87%)
"fake" ██████████ +0.38 ← supports
"virus" ████████ +0.28 ← supports
"lab" ████ -0.13 ← opposes
👥 Team
Name Roll
Name Roll
Madhusudhan G 23MID0444
Shawn Joylin 23MID0185
Aneesh JVR 23MID0306
MDI4001 ML • Winter 2025-26

📜 MIT License
HuggingFace Transformers + LIME + DistilBERT (Sanh et al., 2019)
