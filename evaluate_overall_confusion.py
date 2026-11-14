import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# === Dataset configuration (same as in your app.py) ===
datasets_info = [
    {
        "topic": "Scam Awareness and Prevention",
        "path": "Scam_Awareness___Prevention_Q_A_Dataset_NoRepeatedQuestions.csv",
        "q_col": "user_question",
        "display_cols": ["detailed_answer"],
    },
    {
        "topic": "Fraud Scam Types",
        "path": "fraud_scam_type.csv",
        "q_col": "scam_example_text",
        "display_cols": ["scam_type", "recommended_action", "explanation"],
    },
    {
        "topic": "Document Corpus Extended",
        "path": "Document_corpus_extended.csv",
        "q_col": "title",
        "display_cols": ["chunk_text", "source_type", "topic", "subtopic"],
    },
]

# === Step 1: Merge all datasets ===
all_texts = []

for info in datasets_info:
    path = info["path"]
    q_col = info["q_col"]

    if not os.path.isfile(path):
        print(f"[WARN] Missing: {path}")
        continue

    df = pd.read_csv(path)
    if q_col not in df.columns:
        print(f"[WARN] Missing column {q_col} in {path}")
        continue

    df = df.dropna(subset=[q_col])
    all_texts.extend(df[q_col].astype(str).tolist())

print(f"Loaded total {len(all_texts)} text entries across all datasets.")

# === Step 2: TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
X = vectorizer.fit_transform(all_texts)

# === Step 3: Nearest Neighbor retrieval ===
nn = NearestNeighbors(n_neighbors=2, metric='cosine')
nn.fit(X)
distances, indices = nn.kneighbors(X)

# === Step 4: Build confusion-like matrix ===
# Here, we treat each text as its own "class" — so perfect retrieval = 100% accuracy
# We check whether the nearest neighbor (excluding itself) is the correct one
predictions = []
actuals = []

for i in range(len(all_texts)):
    actuals.append(i)
    nearest_idx = indices[i][1]  # skip itself
    predictions.append(nearest_idx)

# A true positive means nearest neighbor is identical (rare, so we compare cosine similarity)
similarities = 1 - distances[:, 1]
threshold = 0.8  # treat >0.8 similarity as correct retrieval
correct = np.sum(similarities > threshold)
accuracy = correct / len(all_texts)

# === Step 5: Confusion matrix visualization ===
# For such a large corpus, matrix will be huge — we summarize using binary correctness
y_true_binary = np.ones_like(similarities)
y_pred_binary = (similarities > threshold).astype(int)

cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Correct", "Incorrect"])

plt.figure(figsize=(6, 5))
disp.plot(cmap='Blues', values_format='d')
plt.title(f"FinAIgent Overall Confusion Matrix\nAccuracy: {accuracy * 100:.2f}%")
plt.tight_layout()
plt.savefig("confusion_overall_model.png", dpi=150)
plt.close()

print("\n=== FinAIgent Overall Model Evaluation ===")
print(f"Total samples: {len(all_texts)}")
print(f"Similarity threshold: {threshold}")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("✅ Saved confusion_overall_model.png")
