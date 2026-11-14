"""
evaluate_model_graphs.py

Generates multiple diagnostic graphs for the FinAIgent retrieval model (TF-IDF + NearestNeighbors).
Saves PNG files in the current directory.

Requirements:
    pip install scikit-learn pandas matplotlib

Run:
    python evaluate_model_graphs.py
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ----------------------
# Configure datasets (same as app.py)
# ----------------------
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

# ----------------------
# Parameters
# ----------------------
MAX_FEATURES = 20000
TSNE_SAMPLE = 2000   # samples for t-SNE (reduce for memory)
TOP_K = [1, 2, 3, 4, 5]
THRESHOLDS = np.linspace(0.5, 0.95, 10)  # for accuracy vs threshold

# ----------------------
# Load all text entries and track source (which dataset)
# ----------------------
all_texts = []
all_sources = []  # topic name for visualization
for info in datasets_info:
    path = info["path"]
    q_col = info["q_col"]
    if not os.path.isfile(path):
        print(f"[WARN] Missing: {path} — skipping")
        continue
    df = pd.read_csv(path)
    if q_col not in df.columns:
        print(f"[WARN] Missing column {q_col} in {path} — skipping")
        continue
    df = df.dropna(subset=[q_col])
    texts = df[q_col].astype(str).tolist()
    all_texts.extend(texts)
    all_sources.extend([info["topic"]] * len(texts))

if not all_texts:
    raise SystemExit("No text entries loaded. Place CSVs next to this script or update datasets_info paths.")

N = len(all_texts)
print(f"Loaded {N} total text items from {len(set(all_sources))} datasets.")

# ----------------------
# TF-IDF and NearestNeighbors
# ----------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=MAX_FEATURES)
X = vectorizer.fit_transform(all_texts)

nn = NearestNeighbors(n_neighbors=max(max(TOP_K)+1, 6), metric='cosine', n_jobs=-1)
nn.fit(X)
dists, indices = nn.kneighbors(X)  # distances shape: (N, k)
# distances are cosine distances in [0, 2], similarity = 1 - distance (for cosine in sklearn)
# but sklearn's cosine distance ranges [0, 2] when vectors aren't normalized; for TF-IDF it behaves like [0,1]
similarities = 1 - dists  # shape (N, k)

# For top-1 non-self neighbor we use column 1 (col 0 is self)
top1_sim = similarities[:, 1]
top1_idx = indices[:, 1]

# ----------------------
# 1) Similarity distribution histogram
# ----------------------
def plot_similarity_hist(sim_values, out="similarity_histogram.png"):
    plt.figure(figsize=(8, 5))
    plt.hist(sim_values, bins=50)
    plt.xlabel("Cosine similarity to top non-self neighbor")
    plt.ylabel("Count")
    plt.title("Similarity distribution (top non-self neighbor)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

plot_similarity_hist(top1_sim, out="similarity_histogram.png")

# ----------------------
# 2) Accuracy vs threshold
#    For each threshold, count items with top1_sim >= threshold as 'correct'
# ----------------------
def plot_accuracy_vs_threshold(sim_values, thresholds, out="accuracy_vs_threshold.png"):
    accs = []
    for t in thresholds:
        acc = np.mean(sim_values >= t)
        accs.append(acc)
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accs, marker='o')
    plt.xlabel("Similarity threshold")
    plt.ylabel("Fraction considered correct (accuracy)")
    plt.title("Accuracy vs Similarity Threshold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    return thresholds, accs

thr, accs = plot_accuracy_vs_threshold(top1_sim, THRESHOLDS, out="accuracy_vs_threshold.png")

# ----------------------
# 3) Top-K retrieval accuracy
#    We define "correct" if any of top-k neighbors (excluding self) has similarity >= threshold_k
#    For this generic metric we simply check presence of neighbor (i.e., treat retrieval consistency)
# ----------------------
def compute_topk_accuracy(sim_matrix, k_list):
    # sim_matrix shape (N, k_all) where column 0 is self; we consider columns 1..k_all-1
    results = {}
    # for top-k we consider the first k non-self neighbors: columns 1..k
    for k in k_list:
        # check for each row whether any of columns 1..k have similarity >= 0 (they always do) —
        # so instead we measure whether the top-k contains an identical or very-similar item by using a dynamic threshold
        # We'll compute raw top-k hit rate: whether the index of any neighbor is the same "content" (difficult to define).
        # Simpler: measure fraction where top-k max similarity >= median(sim)
        thr = np.median(sim_matrix[:, 1])  # baseline threshold
        hit = np.any(sim_matrix[:, 1:k+1] >= thr, axis=1)
        results[k] = np.mean(hit)
    return results

topk_results = compute_topk_accuracy(similarities, TOP_K)

def plot_topk(topk_dict, out="topk_accuracy.png"):
    xs = sorted(topk_dict.keys())
    ys = [topk_dict[x] for x in xs]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("k (top-k)")
    plt.ylabel("Hit fraction (heuristic)")
    plt.title("Top-k retrieval hit fraction (heuristic baseline)")
    plt.xticks(xs)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

plot_topk(topk_results, out="topk_accuracy.png")
print("Top-k heuristic results:", topk_results)

# ----------------------
# 4) Binary confusion matrix using chosen threshold
# ----------------------
# Choose threshold at a reasonable operating point (you can change)
THRESH = 0.80
y_true = np.ones(N, dtype=int)   # actual = "should be correct"
y_pred = (top1_sim >= THRESH).astype(int)  # predicted correct
cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # [TP, FN; FP, TN] arrangement
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Correct", "Incorrect"])
plt.figure(figsize=(5, 4))
disp.plot(values_format='d')
plt.title(f"Binary Confusion Matrix (threshold={THRESH:.2f})")
plt.tight_layout()
plt.savefig("confusion_overall_binary.png", dpi=150)
plt.close()
print("Saved confusion_overall_binary.png")
accuracy = np.mean(y_pred)
print(f"Binary accuracy (similarity >= {THRESH:.2f} considered correct): {accuracy * 100:.2f}%")

# ----------------------
# 5) PCA -> t-SNE 2D visualization (sampled)
# ----------------------
def plot_tsne_sample(X_sparse, sources, sample=TSNE_SAMPLE, out="tsne_pca.png"):
    n = X_sparse.shape[0]
    if sample < n:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample, replace=False)
    else:
        idx = np.arange(n)
    X_sub = X_sparse[idx].toarray() if hasattr(X_sparse, "toarray") else X_sparse[idx]
    src_sub = [sources[i] for i in idx]

    # PCA to reduce to 50 dims for t-SNE
    pca = PCA(n_components=min(50, X_sub.shape[1]))
    X_pca = pca.fit_transform(X_sub)

    tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=30, n_iter=800)
    X_tsne = tsne.fit_transform(X_pca)

    # Plot
    unique_src = list(sorted(set(src_sub)))
    src_to_int = {s: i for i, s in enumerate(unique_src)}
    labels = [src_to_int[s] for s in src_sub]

    plt.figure(figsize=(10, 8))
    for u in unique_src:
        mask = [s == u for s in src_sub]
        xi = X_tsne[np.array(mask)]
        if xi.size == 0:
            continue
        plt.scatter(xi[:, 0], xi[:, 1], s=8, label=u)
    plt.legend(markerscale=2, fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("t-SNE (PCA->t-SNE) projection of TF-IDF vectors (sampled)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

plot_tsne_sample(X, all_sources, sample=min(TSNE_SAMPLE, N), out="tsne_pca.png")

# ----------------------
# Summary output
# ----------------------
print("\n=== Summary ===")
print(f"Total items: {N}")
print(f"Top-1 similarity (median): {np.median(top1_sim):.4f}, mean: {np.mean(top1_sim):.4f}")
print(f"Fraction with similarity >= {THRESH:.2f}: {np.mean(top1_sim >= THRESH) * 100:.2f}%")
print("Generated plots: similarity_histogram.png, accuracy_vs_threshold.png, topk_accuracy.png,")
print("                 confusion_overall_binary.png, tsne_pca.png")
