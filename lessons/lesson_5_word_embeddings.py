"""
Lesson 5: Word Embeddings
==========================

In Lesson 4, we embedded FLOWERS (numerical data).
Now we embed WORDS (text data).

The challenge: Words are not numbers. How do we represent "cat" as a vector?

Solution 1 (bad):  One-hot encoding - [0,0,0,1,0,0,...]
Solution 2 (good): Word embeddings - learned dense vectors

Key idea: Words that appear in similar CONTEXTS should have similar embeddings.
  "The cat sat on the mat"
  "The dog sat on the rug"
  → "cat" and "dog" appear in similar contexts → similar embeddings
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: The Problem - Words Are Not Numbers
# =============================================================================

print("=" * 60)
print("LESSON 5: WORD EMBEDDINGS")
print("=" * 60)
print()

print("-" * 40)
print("THE PROBLEM: Words Are Not Numbers")
print("-" * 40)
print()

print("""
Neural networks need NUMBERS as input.

Flowers: easy - they're already measurements [5.1, 3.5, 1.4, 0.2]
Words: hard - "cat" is not a number

How do we convert words to numbers?
""")

# =============================================================================
# PART 2: Bad Solution - One-Hot Encoding
# =============================================================================

print("-" * 40)
print("BAD SOLUTION: One-Hot Encoding")
print("-" * 40)
print()

vocabulary = ["king", "queen", "man", "woman", "cat", "dog"]
vocab_size = len(vocabulary)
word_to_idx = {word: i for i, word in enumerate(vocabulary)}

def one_hot(word):
    """Convert word to one-hot vector."""
    vec = np.zeros(vocab_size)
    vec[word_to_idx[word]] = 1
    return vec

print(f"Vocabulary: {vocabulary}")
print(f"Vocabulary size: {vocab_size}")
print()

print("One-hot encodings:")
for word in vocabulary:
    print(f"  {word:6} → {one_hot(word).astype(int)}")
print()

print("Problems with one-hot encoding:")
print("  1. HUGE vectors - real vocabularies have 50,000+ words")
print("  2. NO similarity - every word is equally different from every other")
print()

# Show similarity problem
king_onehot = one_hot("king")
queen_onehot = one_hot("queen")
cat_onehot = one_hot("cat")

def cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / (norm + 1e-8)

print("Cosine similarity with one-hot:")
print(f"  king vs queen: {cosine_sim(king_onehot, queen_onehot):.1f}")
print(f"  king vs cat:   {cosine_sim(king_onehot, cat_onehot):.1f}")
print()
print("Both are 0! One-hot thinks 'king' and 'queen' are as different as 'king' and 'cat'.")
print("That's wrong - 'king' and 'queen' are semantically related!")
print()

# =============================================================================
# PART 3: Good Solution - Word Embeddings
# =============================================================================

print("-" * 40)
print("GOOD SOLUTION: Word Embeddings")
print("-" * 40)
print()

print("""
Instead of sparse one-hot vectors, we use DENSE learned vectors.

One-hot: [0, 0, 0, 1, 0, 0]  (mostly zeros, no meaning)
Embedding: [0.2, 0.8, -0.3, 0.5]  (dense, meaningful)

Each dimension captures some aspect of meaning:
  - Maybe dimension 0 = royalty
  - Maybe dimension 1 = gender
  - etc.

These are LEARNED from data, not hand-designed.
""")

# =============================================================================
# PART 4: How Are Word Embeddings Learned? (Word2Vec Concept)
# =============================================================================

print("-" * 40)
print("HOW ARE EMBEDDINGS LEARNED? (Word2Vec)")
print("-" * 40)
print()

print("""
Core idea: "You shall know a word by the company it keeps."

Training data: sentences
  "The king sat on the throne"
  "The queen sat on the throne"
  "The cat sat on the mat"

Task: Predict a word from its neighbors (or vice versa).

Example - predict middle word from context:
  Input:  "The ___ sat on"
  Target: "king" or "queen" or "cat"

Words that can fill similar blanks get similar embeddings!
  - "king" and "queen" both fit "The ___ sat on the throne"
  - So their embeddings become similar

This is self-supervised learning - no labels needed, just text.
""")

# =============================================================================
# PART 5: Simple Word2Vec Implementation
# =============================================================================

print("-" * 40)
print("SIMPLE WORD2VEC DEMO:")
print("-" * 40)
print()

# Training corpus - simple sentences
corpus = [
    "the king sits on the throne",
    "the queen sits on the throne",
    "the king wears a crown",
    "the queen wears a crown",
    "the man walks in the city",
    "the woman walks in the city",
    "the cat sits on the mat",
    "the dog sits on the rug",
    "the cat chases the mouse",
    "the dog chases the cat",
]

# Build vocabulary
all_words = []
for sentence in corpus:
    all_words.extend(sentence.split())
vocabulary = list(set(all_words))
vocab_size = len(vocabulary)
word_to_idx = {w: i for i, w in enumerate(vocabulary)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

print(f"Corpus: {len(corpus)} sentences")
print(f"Vocabulary: {vocab_size} unique words")
print(f"Words: {vocabulary}")
print()

# Create training pairs: (context_word, target_word)
# Using skip-gram: predict context from center word
def create_training_pairs(corpus, window_size=2):
    pairs = []
    for sentence in corpus:
        words = sentence.split()
        for i, center_word in enumerate(words):
            # Look at words within window
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:
                    context_word = words[j]
                    pairs.append((center_word, context_word))
    return pairs

training_pairs = create_training_pairs(corpus)
print(f"Training pairs: {len(training_pairs)}")
print("Sample pairs (center → context):")
for i in range(10):
    center, context = training_pairs[i]
    print(f"  {center} → {context}")
print()

# =============================================================================
# PART 6: Train Simple Word Embeddings
# =============================================================================

print("-" * 40)
print("TRAINING WORD EMBEDDINGS:")
print("-" * 40)
print()

embedding_dim = 8  # Small for demo

# Initialize embedding matrix: each row is a word's embedding
np.random.seed(42)
W_embed = np.random.randn(vocab_size, embedding_dim) * 0.1  # Input embeddings
W_context = np.random.randn(embedding_dim, vocab_size) * 0.1  # Context weights

def softmax(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

def train_word2vec(training_pairs, W_embed, W_context, lr=0.1, epochs=100):
    """
    Simple Word2Vec training.

    For each (center, context) pair:
    1. Get center word's embedding
    2. Predict probabilities over all words
    3. Loss = -log(prob of correct context word)
    4. Backprop and update
    """
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        np.random.shuffle(training_pairs)

        for center_word, context_word in training_pairs:
            center_idx = word_to_idx[center_word]
            context_idx = word_to_idx[context_word]

            # Forward pass
            # 1. Get center word embedding
            embed = W_embed[center_idx]  # Shape: (embedding_dim,)

            # 2. Compute scores for all words
            scores = embed @ W_context  # Shape: (vocab_size,)

            # 3. Softmax to get probabilities
            probs = softmax(scores)

            # 4. Loss
            loss = -np.log(probs[context_idx] + 1e-8)
            epoch_loss += loss

            # Backward pass
            # Gradient of softmax cross-entropy
            dscores = probs.copy()
            dscores[context_idx] -= 1  # Shape: (vocab_size,)

            # Gradient for W_context
            dW_context = np.outer(embed, dscores)  # Shape: (embedding_dim, vocab_size)

            # Gradient for embedding
            dembed = W_context @ dscores  # Shape: (embedding_dim,)

            # Update
            W_context -= lr * dW_context
            W_embed[center_idx] -= lr * dembed

        avg_loss = epoch_loss / len(training_pairs)
        losses.append(avg_loss)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: loss = {avg_loss:.4f}")

    return W_embed, losses

print(f"Training for 100 epochs...")
print(f"Embedding dimension: {embedding_dim}")
print()

W_embed, losses = train_word2vec(training_pairs, W_embed, W_context)
print()

# =============================================================================
# PART 7: Examine Learned Embeddings
# =============================================================================

print("-" * 40)
print("EXAMINING LEARNED EMBEDDINGS:")
print("-" * 40)
print()

def get_embedding(word):
    return W_embed[word_to_idx[word]]

def most_similar(word, top_k=5):
    """Find words most similar to the given word."""
    word_embed = get_embedding(word)
    similarities = []

    for other_word in vocabulary:
        if other_word != word:
            other_embed = get_embedding(other_word)
            sim = cosine_sim(word_embed, other_embed)
            similarities.append((other_word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Test similarity
test_words = ["king", "cat", "sits"]
for word in test_words:
    print(f"Most similar to '{word}':")
    for similar_word, sim in most_similar(word, top_k=3):
        print(f"  {similar_word}: {sim:.3f}")
    print()

# Compare specific pairs
print("Similarity comparisons:")
pairs = [
    ("king", "queen"),
    ("king", "man"),
    ("cat", "dog"),
    ("king", "cat"),
    ("sits", "walks"),
]
for w1, w2 in pairs:
    sim = cosine_sim(get_embedding(w1), get_embedding(w2))
    print(f"  {w1:6} vs {w2:6}: {sim:.3f}")
print()

# =============================================================================
# PART 8: Visualize Embeddings
# =============================================================================

print("-" * 40)
print("VISUALIZING EMBEDDINGS:")
print("-" * 40)
print()

# Use PCA to reduce to 2D
def simple_pca(X, n_components=2):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    return X_centered @ eigenvectors[:, :n_components]

# Get all embeddings
all_embeddings = np.array([get_embedding(w) for w in vocabulary])
embeddings_2d = simple_pca(all_embeddings)

# Plot
plt.figure(figsize=(10, 8))

# Color code by category
categories = {
    'royalty': ['king', 'queen', 'crown', 'throne'],
    'people': ['man', 'woman'],
    'animals': ['cat', 'dog', 'mouse'],
    'other': []
}

# Assign remaining words to 'other'
categorized = set()
for cat_words in categories.values():
    categorized.update(cat_words)
categories['other'] = [w for w in vocabulary if w not in categorized]

colors = {'royalty': 'gold', 'people': 'blue', 'animals': 'green', 'other': 'gray'}

for cat, words in categories.items():
    for word in words:
        if word in word_to_idx:
            idx = vocabulary.index(word)
            x, y = embeddings_2d[idx]
            plt.scatter(x, y, c=colors[cat], s=100, alpha=0.7, edgecolors='black')
            plt.annotate(word, (x, y), fontsize=12, ha='center', va='bottom')

plt.title('Word Embeddings Visualization\n(Similar words cluster together)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True, alpha=0.3)

# Add legend
for cat, color in colors.items():
    plt.scatter([], [], c=color, label=cat, s=100)
plt.legend()

plt.tight_layout()
plt.savefig('/Users/madhavayyagari/ml-from-scratch/lesson5_word_embeddings.png', dpi=150)
plt.show()

print("Plot saved: lesson5_word_embeddings.png")
print()

# =============================================================================
# PART 9: The Famous Example - Word Arithmetic
# =============================================================================

print("-" * 40)
print("WORD ARITHMETIC:")
print("-" * 40)
print()

print("""
Famous Word2Vec finding: You can do MATH on word meanings!

  king - man + woman ≈ queen

The embedding space captures semantic relationships.
""")

# Try word arithmetic (might not work perfectly with our tiny corpus)
def word_arithmetic(positive, negative, top_k=3):
    """Find word closest to: sum(positive) - sum(negative)"""
    result = np.zeros(embedding_dim)

    for word in positive:
        result += get_embedding(word)
    for word in negative:
        result -= get_embedding(word)

    # Find closest word
    similarities = []
    exclude = set(positive + negative)

    for word in vocabulary:
        if word not in exclude:
            sim = cosine_sim(result, get_embedding(word))
            similarities.append((word, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

print("king - man + woman = ?")
results = word_arithmetic(["king", "woman"], ["man"])
for word, sim in results:
    print(f"  {word}: {sim:.3f}")
print()

print("(With a tiny corpus, results may not be perfect.)")
print("(Real Word2Vec trained on billions of words works much better!)")
print()

# =============================================================================
# PART 10: Summary
# =============================================================================

print("=" * 60)
print("KEY CONCEPTS:")
print("=" * 60)
print("""
1. ONE-HOT vs EMBEDDINGS
   - One-hot: sparse, no similarity info
   - Embeddings: dense, similar words = similar vectors

2. WORD2VEC IDEA
   - "Know a word by its context"
   - Words in similar contexts → similar embeddings
   - Self-supervised: learns from raw text, no labels

3. WHY THIS MATTERS FOR RAG
   - Documents are text → need embeddings
   - Similar documents = similar embeddings
   - Query embedding finds relevant documents

4. IN PRACTICE
   - Use pre-trained embeddings (OpenAI, Sentence-BERT, etc.)
   - They're trained on billions of words
   - Much better than training your own

NEXT: Lesson 6 - Build a simple RAG system!
""")

# Plot loss curve
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Word2Vec Training Loss')
plt.grid(True, alpha=0.3)
plt.savefig('/Users/madhavayyagari/ml-from-scratch/lesson5_training_loss.png', dpi=150)
plt.close()
print("Training loss plot saved: lesson5_training_loss.png")
