"""
Lesson 3: Why Neural Networks Matter (Moons Dataset)
=====================================================

Iris was too easy - a linear classifier got 97%.
Now we use a dataset where LINEAR FAILS COMPLETELY.

The "Moons" dataset: two interleaving crescents.
No straight line can separate them - you NEED curved boundaries.

This lesson shows the real power of neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# =============================================================================
# PART 1: Create the Moons Dataset
# =============================================================================

print("=" * 60)
print("LESSON 3: MOONS DATASET - WHERE LINEAR FAILS")
print("=" * 60)
print()

# Generate moons data
np.random.seed(42)
X, y = make_moons(n_samples=500, noise=0.1)

# Split into train/test
shuffle_idx = np.random.permutation(len(X))
X, y = X[shuffle_idx], y[shuffle_idx]

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]} (2D points)")
print(f"Classes: {len(np.unique(y))} (two moons)")
print()

# Visualize
plt.figure(figsize=(8, 6))
colors = ['blue', 'red']
for c in range(2):
    mask = y_train == c
    plt.scatter(X_train[mask, 0], X_train[mask, 1],
                c=colors[c], label=f'Class {c}', alpha=0.6, edgecolors='black', linewidth=0.5)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Moons Dataset - Can you draw a STRAIGHT line to separate them?')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/Users/madhavayyagari/ml-from-scratch/lesson3_moons_data.png', dpi=150)
plt.show()

print("Plot saved: lesson3_moons_data.png")
print()
print("Look at the plot - NO straight line can separate these classes!")
print()

# =============================================================================
# PART 2: Try Linear Classifier (will fail)
# =============================================================================

print("-" * 60)
print("ATTEMPT 1: LINEAR CLASSIFIER")
print("-" * 60)
print()

def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def train_linear(X_train, y_train, lr=0.5, iterations=500):
    """Train a linear classifier."""
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros(n_classes)

    for i in range(iterations):
        # Forward
        scores = X_train @ W + b
        probs = softmax(scores)

        # Backward
        dscores = probs.copy()
        dscores[np.arange(len(y_train)), y_train] -= 1
        dscores /= len(y_train)

        dW = X_train.T @ dscores
        db = np.sum(dscores, axis=0)

        # Update
        W -= lr * dW
        b -= lr * db

    return W, b

def predict_linear(X, W, b):
    scores = X @ W + b
    return np.argmax(scores, axis=1)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Train linear classifier
np.random.seed(42)
W_linear, b_linear = train_linear(X_train, y_train)

# Evaluate
train_acc_linear = accuracy(y_train, predict_linear(X_train, W_linear, b_linear))
test_acc_linear = accuracy(y_test, predict_linear(X_test, W_linear, b_linear))

print(f"Linear classifier results:")
print(f"  Training accuracy: {train_acc_linear:.1%}")
print(f"  Test accuracy:     {test_acc_linear:.1%}")
print()
print("~85% is the BEST a linear classifier can do on this data.")
print("It's trying to draw a straight line through curved data!")
print()

# =============================================================================
# PART 3: Neural Network (will succeed)
# =============================================================================

print("-" * 60)
print("ATTEMPT 2: NEURAL NETWORK (1 hidden layer)")
print("-" * 60)
print()

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def train_neural_network(X_train, y_train, hidden_size=16, lr=1.0, iterations=1000):
    """Train a 2-layer neural network."""
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    n_samples = len(X_train)

    # Initialize weights
    W1 = np.random.randn(n_features, hidden_size) * 0.5
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, n_classes) * 0.5
    b2 = np.zeros(n_classes)

    losses = []

    for i in range(iterations):
        # Forward pass
        z1 = X_train @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = softmax(z2)

        # Loss
        correct_probs = a2[np.arange(n_samples), y_train]
        loss = -np.mean(np.log(correct_probs + 1e-8))
        losses.append(loss)

        # Backward pass
        dz2 = a2.copy()
        dz2[np.arange(n_samples), y_train] -= 1
        dz2 /= n_samples

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ W2.T
        dz1 = da1 * relu_derivative(z1)

        dW1 = X_train.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if i % 200 == 0:
            preds = np.argmax(a2, axis=1)
            acc = np.mean(preds == y_train)
            print(f"  Iter {i:4d}: loss={loss:.4f}, train_acc={acc:.1%}")

    return W1, b1, W2, b2, losses

def predict_nn(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    return np.argmax(z2, axis=1)

# Train neural network
np.random.seed(42)
print("Training neural network (hidden_size=16)...")
W1, b1, W2, b2, losses = train_neural_network(X_train, y_train, hidden_size=16, lr=1.0, iterations=1000)
print()

# Evaluate
train_acc_nn = accuracy(y_train, predict_nn(X_train, W1, b1, W2, b2))
test_acc_nn = accuracy(y_test, predict_nn(X_test, W1, b1, W2, b2))

print(f"Neural network results:")
print(f"  Training accuracy: {train_acc_nn:.1%}")
print(f"  Test accuracy:     {test_acc_nn:.1%}")
print()

# =============================================================================
# PART 4: Compare visually - Decision Boundaries
# =============================================================================

print("-" * 60)
print("VISUALIZING DECISION BOUNDARIES:")
print("-" * 60)
print()

def plot_decision_boundary(X, y, predict_fn, title, ax):
    """Plot decision boundary for a classifier."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = predict_fn(grid)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    ax.contour(xx, yy, Z, colors='black', linewidths=0.5)

    colors = ['blue', 'red']
    for c in range(2):
        mask = y == c
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[c],
                   alpha=0.6, edgecolors='black', linewidth=0.5)

    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear classifier boundary
plot_decision_boundary(
    X_test, y_test,
    lambda x: predict_linear(x, W_linear, b_linear),
    f'Linear Classifier (Test Acc: {test_acc_linear:.1%})',
    axes[0]
)

# Neural network boundary
plot_decision_boundary(
    X_test, y_test,
    lambda x: predict_nn(x, W1, b1, W2, b2),
    f'Neural Network (Test Acc: {test_acc_nn:.1%})',
    axes[1]
)

plt.tight_layout()
plt.savefig('/Users/madhavayyagari/ml-from-scratch/lesson3_comparison.png', dpi=150)
plt.show()

print("Plot saved: lesson3_comparison.png")
print()

# =============================================================================
# PART 5: Summary
# =============================================================================

print("=" * 60)
print("RESULTS COMPARISON:")
print("=" * 60)
print()
print(f"{'Model':<25} {'Train Acc':<15} {'Test Acc':<15}")
print("-" * 55)
print(f"{'Linear Classifier':<25} {train_acc_linear:<15.1%} {test_acc_linear:<15.1%}")
print(f"{'Neural Network (16 hidden)':<25} {train_acc_nn:<15.1%} {test_acc_nn:<15.1%}")
print()

print("=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("""
1. LINEAR CLASSIFIERS HAVE LIMITS
   - Can only draw straight decision boundaries
   - ~85% is the best possible on moons data
   - No amount of training will fix this

2. NEURAL NETWORKS CAN LEARN CURVES
   - Hidden layer + ReLU enables curved boundaries
   - Achieves 99-100% on the same data
   - The architecture matters, not just more training

3. THIS IS WHY DEEP LEARNING EXISTS
   - Many real problems have non-linear patterns
   - Images, text, audio - all need curved boundaries
   - Adding layers = more complex patterns

4. LOOK AT THE PLOTS!
   - Linear: straight line cutting through both classes
   - Neural net: curved boundary following the moons

NEXT: We'll explore how hidden layers learn "features"
and how this connects to EMBEDDINGS (critical for RAG).
""")
