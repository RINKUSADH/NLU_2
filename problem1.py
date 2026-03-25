import os
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# --- TASK 1: Dataset Preparation ---
print("### TASK 1: Dataset Preparation ###")

# Read the cleaned corpus
documents = []
with open("CleanedCorpus.txt", "r", encoding="utf-8") as f:
    for line in f:
        tokens = line.strip().split()
        if tokens:
            documents.append(tokens)

all_tokens = [token for doc in documents for token in doc]
vocab = set(all_tokens)

print(f"Total number of documents (paragraphs): {len(documents)}")
print(f"Total number of tokens: {len(all_tokens)}")
print(f"Vocabulary size: {len(vocab)}")

# Word Cloud
filtered_tokens = [t for t in all_tokens if t not in STOPWORDS and not t.isdigit() and len(t) > 1]
word_freq = Counter(filtered_tokens)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of IIT Jodhpur Data')
plt.savefig('task1_wordcloud.png')
print("Saved word cloud to task1_wordcloud.png\n")


# --- TASK 2: Model Training ---
print("### TASK 2: Model Training ###")

# We will define a list of param combinations
configs = [
    {"dim": 50, "window": 3, "negative": 3},
    {"dim": 100, "window": 5, "negative": 5},
    {"dim": 150, "window": 5, "negative": 10},
    {"dim": 200, "window": 7, "negative": 5},
    {"dim": 10, "window": 3, "negative": 3},
    {"dim": 20, "window": 5, "negative": 5},
    {"dim": 30, "window": 7, "negative": 10},
    {"dim": 40, "window": 9, "negative": 5}
]

models_cbow = {}
models_sg = {}

with open('report_experiments.txt', 'w') as f:
    f.write("--- Task 2: Word2Vec Model Training Experiments ---\n")
    for cfg in configs:
        dim = cfg["dim"]
        win = cfg["window"]
        neg = cfg["negative"]
        
        # Train CBOW (sg=0)
        cbow = Word2Vec(sentences=documents, vector_size=dim, window=win, min_count=2, sg=0, negative=neg, epochs=50)
        models_cbow[f"dim{dim}_win{win}_neg{neg}"] = cbow
        
        # Train Skip-gram (sg=1)
        sg = Word2Vec(sentences=documents, vector_size=dim, window=win, min_count=2, sg=1, negative=neg, epochs=50)
        models_sg[f"dim{dim}_win{win}_neg{neg}"] = sg
        
        # Calculate loss implicitly (we can just note that training is complete)
        log_str = f"Trained CBOW and Skip-gram with dim={dim}, window={win}, negative={neg}, min_count=2, epochs=50\n"
        print(log_str.strip())
        f.write(log_str)

# We select the central model for further tasks
best_model_name = "dim30_win7_neg10"
cbow_model = models_cbow[best_model_name]
sg_model = models_sg[best_model_name]

print(f"\nProceeding with base configuration ({best_model_name}) for Tasks 3 and 4.")


# --- TASK 3: Semantic Analysis ---
print("\n### TASK 3: Semantic Analysis ###")

target_words = ["research", "student", "phd", "exam"]

with open('report_semantics.txt', 'w') as f:
    f.write("--- Task 3: Nearest Neighbors (CBOW vs SG) ---\n")
    for word in target_words:
        f.write(f"Target word: {word}\n")
        print(f"Top 5 neighbors for '{word}':")
        if word in cbow_model.wv:
            cbow_nn = cbow_model.wv.most_similar(word, topn=5)
            # sg_nn
            sg_nn = sg_model.wv.most_similar(word, topn=5)
            f.write(f"CBOW: {cbow_nn}\n")
            f.write(f"Skip-gram: {sg_nn}\n\n")
            print(f"  CBOW: {[x[0] for x in cbow_nn]}")
            print(f"  SG:   {[x[0] for x in sg_nn]}")
        else:
            f.write(f"Word '{word}' not in vocabulary.\n")

    f.write("\n--- Analogies ---\n")
    print("\nAnalogies:")
    analogies = [
        ("ug", "btech", "pg"),
        ("phd", "research", "exam"), 
        ("student", "exam", "faculty"), 
        ("cse", "programming", "mechanical"),
        ("student", "hostel", "faculty")
    ]
    
    ground_truth = {
        ("ug", "btech", "pg"): ["mtech", "msc", "phd", "postgraduate", "mba", "masters", "pg"],
        ("phd", "research", "exam"): ["study", "grade", "marks", "pass", "fail", "score", "test", "coursework", "degree"],
        ("student", "exam", "faculty"): ["teaching", "grading", "evaluation", "teach", "research", "course", "paper", "assessment"],
        ("cse", "programming", "mechanical"): ["manufacturing", "machines", "machine", "workshop", "cad", "mechanics", "robotics", "automobile", "design", "core"],
        ("student", "hostel", "faculty"): ["quarters", "housing", "office", "residence", "accommodation", "qtrs", "home", "staff", "department"]
    }

    for w1, w2, w3 in analogies:
        if w1 in cbow_model.wv and w2 in cbow_model.wv and w3 in cbow_model.wv:
            cbow_res = cbow_model.wv.most_similar(positive=[w2, w3], negative=[w1], topn=1)
            sg_res = sg_model.wv.most_similar(positive=[w2, w3], negative=[w1], topn=1)
            msg = f"{w1} : {w2} :: {w3} : ? \n  -> CBOW: {cbow_res[0][0]:<15} (cosine score: {cbow_res[0][1]:.4f})\n  -> SG:   {sg_res[0][0]:<15} (cosine score: {sg_res[0][1]:.4f})"
            print(msg)
            f.write(msg + "\n")
        else:
            msg = f"Skipping analogy {w1}:{w2}::{w3} due to missing vocab."
            print(msg)
            f.write(msg + "\n")

    print("\nEvaluating Intrinsic Analogy Confidence/Accuracy metrics across ALL Task 2 models...")
    f.write("\n--- All Task 2 Models Analogy Confidence/Accuracy Metrics ---\n")
    
    def evaluate_model_accuracy(model, analogies, ground_truth):
        correct = 0
        total = 0
        avg_sim = 0.0
        for w1, w2, w3 in analogies:
            if w1 in model.wv and w2 in model.wv and w3 in model.wv:
                total += 1
                res = model.wv.most_similar(positive=[w2, w3], negative=[w1], topn=5)
                avg_sim += res[0][1] # similarity of top-1
                predicted_words = [r[0].lower() for r in res]
                gt_words = ground_truth.get((w1, w2, w3), [])
                if any(gt in predicted_words for gt in gt_words):
                    correct += 1
        acc = (correct / total * 100) if total > 0 else 0
        mean_sim = (avg_sim / total) if total > 0 else 0
        return acc, mean_sim, total

    print(f"{'Model Name':<25} | {'Type':<5} | {'Acc (Top-5 vs GT)':<18} | {'Avg Top-1 Cosine Sim':<20}")
    f.write(f"{'Model Name':<25} | {'Type':<5} | {'Acc (Top-5 vs GT)':<18} | {'Avg Top-1 Cosine Sim':<20}\n")
    f.write("-" * 75 + "\n")
    print("-" * 75)
    
    cbow_scores = {}
    sg_scores = {}

    for name, model in models_cbow.items():
        acc, sim, valid = evaluate_model_accuracy(model, analogies, ground_truth)
        cbow_scores[name] = sim
        msg = f"{name:<25} | CBOW  | {acc:>6.2f}% ({valid} valid)    | {sim:.4f}"
        print(msg)
        f.write(msg + "\n")
        
    for name, model in models_sg.items():
        acc, sim, valid = evaluate_model_accuracy(model, analogies, ground_truth)
        sg_scores[name] = sim
        msg = f"{name:<25} | SG    | {acc:>6.2f}% ({valid} valid)    | {sim:.4f}"
        print(msg)
        f.write(msg + "\n")

    mean_cbow_sim = sum(cbow_scores.values()) / len(cbow_scores) if cbow_scores else 0
    mean_sg_sim = sum(sg_scores.values()) / len(sg_scores) if sg_scores else 0
    
    print("\n--- Model Performance Comparison ---")
    f.write("\n--- Model Performance Comparison ---\n")
    if mean_cbow_sim > mean_sg_sim:
        better_msg = f"Overall, CBOW performs better than Skip-gram in generating confident analogy predictions (Avg Top-1 Sim: CBOW {mean_cbow_sim:.4f} > SG {mean_sg_sim:.4f})."
    else:
        better_msg = f"Overall, Skip-gram performs better than CBOW in generating confident analogy predictions (Avg Top-1 Sim: SG {mean_sg_sim:.4f} > CBOW {mean_cbow_sim:.4f})."
    
    print(better_msg)
    f.write(better_msg + "\n")

# --- TASK 4: Visualization ---
print("\n### TASK 4: Visualization ###")
words_to_plot = target_words + ["ug", "pg", "btech", "mtech", "faculty", "program", "department", "academic"]
valid_words = [w for w in words_to_plot if w in cbow_model.wv]

if valid_words:
    # Get vectors
    cbow_vectors = np.array([cbow_model.wv[w] for w in valid_words])
    sg_vectors = np.array([sg_model.wv[w] for w in valid_words])

    # Perform PCA
    pca_cbow = PCA(n_components=2).fit_transform(cbow_vectors)
    pca_sg = PCA(n_components=2).fit_transform(sg_vectors)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(pca_cbow[:, 0], pca_cbow[:, 1], c='blue')
    for i, word in enumerate(valid_words):
        plt.annotate(word, (pca_cbow[i, 0], pca_cbow[i, 1]))
    plt.title('CBOW Embeddings (PCA)')

    plt.subplot(1, 2, 2)
    plt.scatter(pca_sg[:, 0], pca_sg[:, 1], c='red')
    for i, word in enumerate(valid_words):
        plt.annotate(word, (pca_sg[i, 0], pca_sg[i, 1]))
    plt.title('Skip-gram Embeddings (PCA)')

    plt.savefig('task4_clusters.png')
    print("Saved PCA cluster visualization to task4_clusters.png")

    # Perform TSNE
    perplexity_val = min(5, len(cbow_vectors) - 1)
    if perplexity_val > 0:
        tsne_cbow = TSNE(n_components=2, random_state=42, perplexity=perplexity_val).fit_transform(cbow_vectors)
        tsne_sg = TSNE(n_components=2, random_state=42, perplexity=perplexity_val).fit_transform(sg_vectors)

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(tsne_cbow[:, 0], tsne_cbow[:, 1], c='blue')
        for i, word in enumerate(valid_words):
            plt.annotate(word, (tsne_cbow[i, 0], tsne_cbow[i, 1]))
        plt.title('CBOW Embeddings (t-SNE)')

        plt.subplot(1, 2, 2)
        plt.scatter(tsne_sg[:, 0], tsne_sg[:, 1], c='red')
        for i, word in enumerate(valid_words):
            plt.annotate(word, (tsne_sg[i, 0], tsne_sg[i, 1]))
        plt.title('Skip-gram Embeddings (t-SNE)')

        plt.savefig('task4_tsne_clusters.png')
        print("Saved t-SNE cluster visualization to task4_tsne_clusters.png")

print("\nProblem 1 completed successfully.")
