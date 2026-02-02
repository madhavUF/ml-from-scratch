"""
Lesson 2: From Linear Classifier to Neural Network
===================================================

In Lesson 1, we built:
    output = softmax(X @ W + b)

This is LIMITED - it can only draw straight decision boundaries.

Now we add a HIDDEN LAYER:
    hidden = ReLU(X @ W1 + b1)      # Layer 1: extract features
    output = softmax(hidden @ W2 + b2)  # Layer 2: classify

This is a NEURAL NETWORK - it can learn curved boundaries.

KEY NEW CONCEPTS:
1. Hidden layer - intermediate representation
2. ReLU activation - introduces non-linearity
3. Backpropagation - gradients through multiple layers
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: Why do we need hidden layers?
# =============================================================================

print("=" * 60)
print("LESSON 2: NEURAL NETWORKS")
print("=" * 60)
print()

print("-" * 40)
print("WHY HIDDEN LAYERS?")
print("-" * 40)
print("""
Linear classifier: output = X @ W + b
  → Can only draw STRAIGHT lines between classes
  → Fails on data that isn't linearly separable

Neural network:
  hidden = ReLU(X @ W1 + b1)
  output = hidden @ W2 + b2
  → Can draw CURVED boundaries
  → Can solve more complex problems

The hidden layer learns FEATURES from raw input.
Then the output layer classifies based on those features.
""")

# =============================================================================
# PART 2: What is ReLU?
# =============================================================================

print("-" * 40)
print("ReLU (Rectified Linear Unit):")
print("-" * 40)
print()

print("ReLU(x) = max(0, x)")
print("  - If x > 0: output = x")
print("  - If x ≤ 0: output = 0")
print()

# Visualize ReLU
x = np.linspace(-3, 3, 100)
relu_x = np.maximum(0, x)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x, relu_x, 'b-', linewidth=2)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU: max(0, x)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)

# Why ReLU matters - show linear vs nonlinear
plt.subplot(1, 2, 2)
plt.plot(x, x, 'r--', label='Linear (no activation)', linewidth=2)
plt.plot(x, relu_x, 'b-', label='ReLU (nonlinear)', linewidth=2)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Linear vs Nonlinear')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/madhavayyagari/ml-from-scratch/lesson2_relu.png', dpi=150)
plt.show()

print("Plot saved: lesson2_relu.png")
print()

print("""
WHY ReLU?
- Without it: hidden = X @ W1, output = hidden @ W2
  This simplifies to: output = X @ (W1 @ W2) = X @ W_combined
  → Just another linear classifier! Useless.

- With ReLU: The nonlinearity lets the network learn
  complex patterns that linear models can't.
""")

# =============================================================================
# PART 3: Load the Iris data
# =============================================================================

print("-" * 40)
print("LOADING DATA:")
print("-" * 40)
print()

data = np.load('/Users/madhavayyagari/ml-from-scratch/iris_prepared.npz',
               allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
target_names = data['target_names']

num_features = X_train.shape[1]  # 4
num_classes = len(np.unique(y_train))  # 3
num_samples = len(X_train)

print(f"Training: {num_samples} samples")
print(f"Features: {num_features}")
print(f"Classes: {num_classes}")
print()

# =============================================================================
# PART 4: Define the Neural Network
# =============================================================================

print("-" * 40)
print("NEURAL NETWORK ARCHITECTURE:")
print("-" * 40)
print()

hidden_size = 10  # Number of neurons in hidden layer

print(f"Input layer:  {num_features} neurons (features)")
print(f"Hidden layer: {hidden_size} neurons")
print(f"Output layer: {num_classes} neurons (classes)")
print()
print("Architecture: 4 → 10 → 3")
print()

# Initialize weights
np.random.seed(42)

# Layer 1: input → hidden
W1 = np.random.randn(num_features, hidden_size) * 0.1
b1 = np.zeros(hidden_size)

# Layer 2: hidden → output
W2 = np.random.randn(hidden_size, num_classes) * 0.1
b2 = np.zeros(num_classes)

total_params = W1.size + b1.size + W2.size + b2.size
print(f"Parameters:")
print(f"  W1: {W1.shape} = {W1.size}")
print(f"  b1: {b1.shape} = {b1.size}")
print(f"  W2: {W2.shape} = {W2.size}")
print(f"  b2: {b2.shape} = {b2.size}")
print(f"  Total: {total_params}")
print()

# =============================================================================
# PART 5: Forward Pass
# =============================================================================

print("-" * 40)
print("FORWARD PASS:")
print("-" * 40)
print()

def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def softmax(x):
    """Softmax: convert scores to probabilities"""
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    """
    Forward pass through 2-layer neural network.

    Returns all intermediate values (needed for backprop).
    """
    # Layer 1: Linear + ReLU
    z1 = X @ W1 + b1           # Linear transformation
    a1 = relu(z1)              # ReLU activation

    # Layer 2: Linear + Softmax
    z2 = a1 @ W2 + b2          # Linear transformation
    a2 = softmax(z2)           # Softmax activation

    # Return everything (needed for backward pass)
    cache = (X, z1, a1, z2, a2)
    return a2, cache

# Test forward pass
probs, cache = forward(X_train, W1, b1, W2, b2)
print("Forward pass shapes:")
print(f"  Input X:     {X_train.shape}")
print(f"  Hidden z1:   {cache[1].shape}")
print(f"  Hidden a1:   {cache[2].shape} (after ReLU)")
print(f"  Output z2:   {cache[3].shape}")
print(f"  Output a2:   {cache[4].shape} (probabilities)")
print()

# =============================================================================
# PART 6: Loss Function (same as before)
# =============================================================================

def compute_loss(probs, y):
    """Cross-entropy loss."""
    correct_probs = probs[np.arange(len(y)), y]
    return -np.mean(np.log(correct_probs + 1e-8))

loss = compute_loss(probs, y_train)
print(f"Initial loss (random weights): {loss:.4f}")
print()

# =============================================================================
# PART 7: Backward Pass (Backpropagation)
# =============================================================================

print("-" * 40)
print("BACKWARD PASS (Backpropagation):")
print("-" * 40)
print()

print("""
Backpropagation = chain rule applied layer by layer.

We compute gradients BACKWARDS from output to input:
1. dL/dz2 (output layer gradient)
2. dL/dW2, dL/db2
3. dL/da1 (gradient flowing back to hidden layer)
4. dL/dz1 (through ReLU)
5. dL/dW1, dL/db1

The key insight: gradient flows backward through each operation.
""")

def relu_derivative(z):
    """
    Derivative of ReLU.
    ReLU(z) = max(0, z)
    d/dz ReLU(z) = 1 if z > 0, else 0
    """
    return (z > 0).astype(float)

def backward(y, cache, W1, b1, W2, b2):
    """
    Backward pass: compute gradients for all parameters.
    """
    X, z1, a1, z2, a2 = cache
    num_samples = len(y)

    # === Output layer gradients ===
    # dL/dz2 = a2 - one_hot(y)  (same as lesson 1)
    dz2 = a2.copy()
    dz2[np.arange(num_samples), y] -= 1
    dz2 /= num_samples

    # dL/dW2 = a1.T @ dz2
    dW2 = a1.T @ dz2

    # dL/db2 = sum(dz2)
    db2 = np.sum(dz2, axis=0)

    # === Hidden layer gradients ===
    # Gradient flows back through W2
    # dL/da1 = dz2 @ W2.T
    da1 = dz2 @ W2.T

    # Gradient flows back through ReLU
    # dL/dz1 = dL/da1 * ReLU'(z1)
    dz1 = da1 * relu_derivative(z1)

    # dL/dW1 = X.T @ dz1
    dW1 = X.T @ dz1

    # dL/db1 = sum(dz1)
    db1 = np.sum(dz1, axis=0)

    return dW1, db1, dW2, db2

# Test backward pass
dW1, db1, dW2, db2 = backward(y_train, cache, W1, b1, W2, b2)
print("Gradient shapes:")
print(f"  dW1: {dW1.shape} (same as W1)")
print(f"  db1: {db1.shape} (same as b1)")
print(f"  dW2: {dW2.shape} (same as W2)")
print(f"  db2: {db2.shape} (same as b2)")
print()

# =============================================================================
# PART 8: Training Loop
# =============================================================================

print("-" * 40)
print("TRAINING:")
print("-" * 40)
print()

# Re-initialize weights
np.random.seed(42)
W1 = np.random.randn(num_features, hidden_size) * 0.1
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, num_classes) * 0.1
b2 = np.zeros(num_classes)

# Hyperparameters
learning_rate = 0.5
num_iterations = 500

# Track history
train_losses = []
train_accuracies = []
test_accuracies = []

def predict(X, W1, b1, W2, b2):
    probs, _ = forward(X, W1, b1, W2, b2)
    return np.argmax(probs, axis=1)

def accuracy(X, y, W1, b1, W2, b2):
    return np.mean(predict(X, W1, b1, W2, b2) == y)

print(f"Learning rate: {learning_rate}")
print(f"Iterations: {num_iterations}")
print(f"Hidden size: {hidden_size}")
print()
print(f"{'Iter':<6} {'Loss':<10} {'Train Acc':<12} {'Test Acc':<12}")
print("-" * 40)

for i in range(num_iterations):
    # Forward
    probs, cache = forward(X_train, W1, b1, W2, b2)

    # Loss
    loss = compute_loss(probs, y_train)
    train_losses.append(loss)

    # Backward
    dW1, db1, dW2, db2 = backward(y_train, cache, W1, b1, W2, b2)

    # Update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Track accuracy
    train_acc = accuracy(X_train, y_train, W1, b1, W2, b2)
    test_acc = accuracy(X_test, y_test, W1, b1, W2, b2)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    if i % 50 == 0 or i == num_iterations - 1:
        print(f"{i:<6} {loss:<10.4f} {train_acc:<12.1%} {test_acc:<12.1%}")

print("-" * 40)
print()

# =============================================================================
# PART 9: Results
# =============================================================================

print("-" * 40)
print("FINAL RESULTS:")
print("-" * 40)
print()

final_train_acc = accuracy(X_train, y_train, W1, b1, W2, b2)
final_test_acc = accuracy(X_test, y_test, W1, b1, W2, b2)

print(f"Training accuracy: {final_train_acc:.1%}")
print(f"Test accuracy:     {final_test_acc:.1%}")
print()

# Compare with Lesson 1 linear classifier
print("Comparison with Lesson 1 (linear classifier):")
print("  Linear classifier test accuracy: ~97%")
print(f"  Neural network test accuracy:    {final_test_acc:.0%}")
print()
print("(On Iris, linear is sufficient - but neural nets shine on")
print(" harder problems where linear boundaries don't work.)")
print()

# =============================================================================
# PART 10: Visualize
# =============================================================================

print("-" * 40)
print("GENERATING PLOTS...")
print("-" * 40)
print()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Loss curve
axes[0].plot(train_losses, 'b-', linewidth=2)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(train_accuracies, 'b-', label='Train', linewidth=2)
axes[1].plot(test_accuracies, 'r--', label='Test', linewidth=2)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Over Training')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1.05])

# Hidden layer activations for test set
probs, cache = forward(X_test, W1, b1, W2, b2)
hidden_activations = cache[2]  # a1

# Show first 2 hidden neurons as a scatter plot
colors = ['purple', 'green', 'orange']
for c in range(num_classes):
    mask = y_test == c
    axes[2].scatter(hidden_activations[mask, 0],
                    hidden_activations[mask, 1],
                    c=colors[c], label=target_names[c],
                    alpha=0.7, edgecolors='black', linewidth=0.5)
axes[2].set_xlabel('Hidden Neuron 1')
axes[2].set_ylabel('Hidden Neuron 2')
axes[2].set_title('Learned Representations (Hidden Layer)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/madhavayyagari/ml-from-scratch/lesson2_results.png', dpi=150)
plt.show()

print("Plot saved: lesson2_results.png")
print()

# =============================================================================
# KEY CONCEPTS
# =============================================================================

print("=" * 60)
print("LESSON 2 COMPLETE - KEY CONCEPTS:")
print("=" * 60)
print("""
1. HIDDEN LAYER
   - Sits between input and output
   - Learns intermediate features/representations
   - More hidden neurons = more capacity to learn

2. ReLU ACTIVATION
   - ReLU(x) = max(0, x)
   - Makes network NONLINEAR
   - Without it, multiple layers collapse to one

3. BACKPROPAGATION
   - Chain rule applied backwards through layers
   - Each layer passes gradient to the previous layer
   - dz1 = (dz2 @ W2.T) * relu_derivative(z1)

4. NETWORK ARCHITECTURE: 4 → 10 → 3
   - Input: 4 features
   - Hidden: 10 neurons (hyperparameter you choose)
   - Output: 3 classes

THE TRAINING LOOP (now with 2 layers):
1. Forward: X → z1 → a1 → z2 → a2 (probs)
2. Loss: cross-entropy
3. Backward: dW2, db2, dW1, db1
4. Update: all 4 parameter matrices

WHAT'S NEXT?
- Lesson 3: Deeper networks, batch training, regularization
- Eventually: Embeddings, which lead to RAG systems
""")

# Save model
np.savez('/Users/madhavayyagari/ml-from-scratch/model_nn_trained.npz',
         W1=W1, b1=b1, W2=W2, b2=b2)
print("Model saved to model_nn_trained.npz")
