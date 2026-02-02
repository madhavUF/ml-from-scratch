"""
Lesson 4: Embeddings - The Key to RAG
======================================

Remember in Lesson 2, the hidden layer learned to separate flowers?
That hidden layer output IS an embedding.

EMBEDDING = a learned vector representation of something
           (a word, a sentence, an image, etc.)

Key insight:
- Similar things → similar embeddings (close in vector space)
- Different things → different embeddings (far apart)

This is THE foundation of RAG:
- Convert documents to embeddings
- Convert query to embedding
- Find documents with similar embeddings
- Feed those documents to the LLM
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: What is an Embedding?
# =============================================================================

print("=" * 60)
print("LESSON 4: EMBEDDINGS")
print("=" * 60)
print()

print("-" * 40)
print("WHAT IS AN EMBEDDING?")
print("-" * 40)
print("""
An embedding is a VECTOR (list of numbers) that represents something.

Examples:
  "king"  → [0.2, 0.8, 0.1, 0.9, ...]   (maybe 300 numbers)
  "queen" → [0.2, 0.7, 0.1, 0.9, ...]   (similar to king!)
  "apple" → [0.9, 0.1, 0.8, 0.2, ...]   (very different)

The MAGIC: These numbers are LEARNED so that:
  - Similar meanings → similar vectors
  - Different meanings → different vectors

This lets us do MATH on meaning:
  king - man + woman ≈ queen
""")

# =============================================================================
# PART 2: You Already Made Embeddings!
# =============================================================================

print("-" * 40)
print("YOU ALREADY MADE EMBEDDINGS!")
print("-" * 40)
print()

print("Remember Lesson 2's neural network?")
print()
print("  Input (4 features) → Hidden (10 neurons) → Output (3 classes)")
print()
print("The hidden layer output IS an embedding!")
print("  - Input: raw measurements [5.1, 3.5, 1.4, 0.2]")
print("  - Hidden: learned representation [0.2, 0.0, 0.8, ...]")
print("  - Output: class probabilities [0.95, 0.03, 0.02]")
print()
print("The network learned to transform raw features into a space")
print("where different flowers are easy to separate.")
print()

# Load the trained neural network from Lesson 2
try:
    model = np.load('/Users/madhavayyagari/ml-from-scratch/model_nn_trained.npz')
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    print("Loaded trained model from Lesson 2")
except:
    print("Training a quick model...")
    # Quick train if model doesn't exist
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = (iris.data - iris.data.mean(0)) / iris.data.std(0)
    y = iris.target

    np.random.seed(42)
    W1 = np.random.randn(4, 10) * 0.1
    b1 = np.zeros(10)
    W2 = np.random.randn(10, 3) * 0.1
    b2 = np.zeros(3)

    for _ in range(500):
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2 + b2
        exp_z2 = np.exp(z2 - z2.max(1, keepdims=True))
        probs = exp_z2 / exp_z2.sum(1, keepdims=True)

        dz2 = probs.copy()
        dz2[np.arange(len(y)), y] -= 1
        dz2 /= len(y)

        dW2 = a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(0)

        W1 -= 0.5 * dW1
        b1 -= 0.5 * db1
        W2 -= 0.5 * dW2
        b2 -= 0.5 * db2

# Load iris data
data = np.load('/Users/madhavayyagari/ml-from-scratch/iris_prepared.npz', allow_pickle=True)
X_test = data['X_test']
y_test = data['y_test']
target_names = data['target_names']

# =============================================================================
# PART 3: Extract Embeddings
# =============================================================================

print()
print("-" * 40)
print("EXTRACTING EMBEDDINGS:")
print("-" * 40)
print()

def get_embedding(X, W1, b1):
    """
    Get the hidden layer representation (embedding).
    This is just the first half of the forward pass.
    """
    z1 = X @ W1 + b1
    embedding = np.maximum(0, z1)  # ReLU
    return embedding

# Get embeddings for test set
embeddings = get_embedding(X_test, W1, b1)

print(f"Input shape:     {X_test.shape}     (30 flowers, 4 raw features)")
print(f"Embedding shape: {embeddings.shape}  (30 flowers, 10-dim embedding)")
print()

print("Example - Flower #0:")
print(f"  Raw input:  {X_test[0].round(2)}")
print(f"  Embedding:  {embeddings[0].round(2)}")
print(f"  True class: {target_names[y_test[0]]}")
print()

# =============================================================================
# PART 4: Similar Things Have Similar Embeddings
# =============================================================================

print("-" * 40)
print("SIMILAR THINGS → SIMILAR EMBEDDINGS:")
print("-" * 40)
print()

def cosine_similarity(a, b):
    """
    Cosine similarity: how similar are two vectors?
    Returns value between -1 and 1.
      1 = identical direction
      0 = perpendicular (unrelated)
     -1 = opposite direction
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b + 1e-8)

# Find one flower of each type
idx_setosa = np.where(y_test == 0)[0][0]
idx_versicolor = np.where(y_test == 1)[0][0]
idx_virginica = np.where(y_test == 2)[0][0]

# Another setosa for comparison
idx_setosa2 = np.where(y_test == 0)[0][1]

emb_setosa = embeddings[idx_setosa]
emb_setosa2 = embeddings[idx_setosa2]
emb_versicolor = embeddings[idx_versicolor]
emb_virginica = embeddings[idx_virginica]

print("Cosine similarity between embeddings:")
print()
print(f"  Setosa vs Setosa (same class):     {cosine_similarity(emb_setosa, emb_setosa2):.3f}")
print(f"  Setosa vs Versicolor (different):  {cosine_similarity(emb_setosa, emb_versicolor):.3f}")
print(f"  Setosa vs Virginica (different):   {cosine_similarity(emb_setosa, emb_virginica):.3f}")
print(f"  Versicolor vs Virginica (both not setosa): {cosine_similarity(emb_versicolor, emb_virginica):.3f}")
print()
print("Notice: Same class = HIGH similarity, different class = LOWER similarity")
print()

# =============================================================================
# PART 5: Visualize Embeddings
# =============================================================================

print("-" * 40)
print("VISUALIZING EMBEDDINGS:")
print("-" * 40)
print()

# Use PCA to reduce 10D embeddings to 2D for visualization
def simple_pca(X, n_components=2):
    """Simple PCA: project high-dim data to lower dimensions."""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    return X_centered @ eigenvectors[:, :n_components]

# Project embeddings to 2D
embeddings_2d = simple_pca(embeddings, n_components=2)

# Also project raw inputs to 2D for comparison
raw_2d = simple_pca(X_test, n_components=2)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = ['purple', 'green', 'orange']

# Raw input space
for c in range(3):
    mask = y_test == c
    axes[0].scatter(raw_2d[mask, 0], raw_2d[mask, 1],
                    c=colors[c], label=target_names[c],
                    alpha=0.7, edgecolors='black', linewidth=0.5, s=80)
axes[0].set_title('Raw Input Space (PCA of 4 features)')
axes[0].set_xlabel('PC 1')
axes[0].set_ylabel('PC 2')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Embedding space
for c in range(3):
    mask = y_test == c
    axes[1].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=colors[c], label=target_names[c],
                    alpha=0.7, edgecolors='black', linewidth=0.5, s=80)
axes[1].set_title('Embedding Space (PCA of 10-dim hidden layer)')
axes[1].set_xlabel('PC 1')
axes[1].set_ylabel('PC 2')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/madhavayyagari/ml-from-scratch/lesson4_embeddings.png', dpi=150)
plt.show()

print("Plot saved: lesson4_embeddings.png")
print()
print("Look at the plot:")
print("  - Left: Raw features - classes overlap somewhat")
print("  - Right: Embeddings - classes are CLEARLY separated!")
print()
print("The neural network LEARNED to transform data into a space")
print("where similar things cluster together.")
print()

# =============================================================================
# PART 6: Semantic Search (mini RAG demo)
# =============================================================================

print("-" * 40)
print("SEMANTIC SEARCH (Mini RAG Demo):")
print("-" * 40)
print()

print("""
This is how RAG retrieval works:

1. You have a "database" of items (documents, flowers, etc.)
2. Each item has an embedding
3. Given a query, compute its embedding
4. Find items with most similar embeddings
5. Return those items

Let's try it with our flowers!
""")

# Our "database" of flowers
database_embeddings = embeddings
database_labels = y_test

def search(query_embedding, database_embeddings, top_k=3):
    """Find the top_k most similar items in the database."""
    similarities = []
    for i, db_emb in enumerate(database_embeddings):
        sim = cosine_similarity(query_embedding, db_emb)
        similarities.append((i, sim))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Query: a new flower (let's use one from the test set as if it's new)
query_idx = 15
query_input = X_test[query_idx]
query_embedding = get_embedding(query_input.reshape(1, -1), W1, b1)[0]
query_true_class = target_names[y_test[query_idx]]

print(f"Query flower (true class: {query_true_class}):")
print(f"  Raw features: {query_input.round(2)}")
print()

# Search
results = search(query_embedding, database_embeddings, top_k=5)

print("Top 5 most similar flowers in database:")
print(f"{'Rank':<6} {'Index':<8} {'Class':<15} {'Similarity':<12}")
print("-" * 45)
for rank, (idx, sim) in enumerate(results, 1):
    class_name = target_names[y_test[idx]]
    print(f"{rank:<6} {idx:<8} {class_name:<15} {sim:<12.3f}")
print()

# Check if retrieval is correct
retrieved_classes = [target_names[y_test[idx]] for idx, _ in results]
print(f"Query class: {query_true_class}")
print(f"Retrieved classes: {retrieved_classes}")
if retrieved_classes[0] == query_true_class:
    print("✓ Top result is correct class!")
print()

# =============================================================================
# PART 7: Connection to RAG
# =============================================================================

print("=" * 60)
print("HOW THIS CONNECTS TO RAG:")
print("=" * 60)
print("""
What we just did with flowers is EXACTLY what RAG does with text:

FLOWERS (what we did):
  1. Raw input: [5.1, 3.5, 1.4, 0.2]  (measurements)
  2. Embedding: neural net hidden layer
  3. Search: find similar flowers by cosine similarity

TEXT (what RAG does):
  1. Raw input: "What is machine learning?"  (text)
  2. Embedding: transformer model (like BERT, OpenAI embeddings)
  3. Search: find similar documents by cosine similarity

The process is IDENTICAL:
  query → embedding → find similar embeddings → return results

RAG then takes those retrieved documents and feeds them to an LLM:
  "Here are relevant documents: [...]
   Based on these, answer: What is machine learning?"
""")

# =============================================================================
# PART 8: Summary
# =============================================================================

print("=" * 60)
print("KEY CONCEPTS:")
print("=" * 60)
print("""
1. EMBEDDING = vector representation
   - Raw data → neural network → embedding (hidden layer)
   - Similar things get similar embeddings

2. COSINE SIMILARITY
   - Measures angle between vectors
   - 1 = same direction (similar)
   - 0 = perpendicular (unrelated)

3. SEMANTIC SEARCH
   - Convert query to embedding
   - Find nearest embeddings in database
   - This is the "R" (Retrieval) in RAG

4. THE NEURAL NET'S JOB
   - Learn embeddings where similar = close, different = far
   - The hidden layer IS the embedding

WHAT'S NEXT:
- Lesson 5: Word embeddings (Word2Vec concept)
- Lesson 6: Simple RAG system
""")
