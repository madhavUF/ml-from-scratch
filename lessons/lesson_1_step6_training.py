"""
Lesson 1, Step 6: The Training Loop
====================================

Now we put it all together!

The training loop repeats:
1. FORWARD: Compute predictions
2. LOSS: Measure how wrong we are
3. BACKWARD: Compute gradients
4. UPDATE: Adjust weights to reduce loss

This is ALL of machine learning training in a nutshell.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# STEP 6A: Load everything
# =============================================================================

print("=" * 60)
print("STEP 6: THE TRAINING LOOP")
print("=" * 60)
print()

# Load data
data = np.load('/Users/madhavayyagari/ml-from-scratch/iris_prepared.npz',
               allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
target_names = data['target_names']

# Initialize fresh weights
np.random.seed(42)
num_features = 4
num_classes = 3
W = np.random.randn(num_features, num_classes) * 0.01
b = np.zeros(num_classes)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Parameters: {W.size + b.size} ({W.size} weights + {b.size} biases)")
print()

# =============================================================================
# STEP 6B: Define all our helper functions
# =============================================================================

def softmax(scores):
    """Convert scores to probabilities."""
    scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def forward(X, W, b):
    """Forward pass: compute probabilities."""
    scores = X @ W + b
    probs = softmax(scores)
    return probs


def compute_loss(probs, y):
    """Compute cross-entropy loss."""
    correct_probs = probs[np.arange(len(y)), y]
    return -np.mean(np.log(correct_probs + 1e-8))


def compute_gradients(X, y, probs):
    """Compute gradients of loss w.r.t. W and b."""
    num_samples = len(y)
    dscores = probs.copy()
    dscores[np.arange(num_samples), y] -= 1
    dscores /= num_samples
    dW = X.T @ dscores
    db = np.sum(dscores, axis=0)
    return dW, db


def predict(X, W, b):
    """Return predicted class labels."""
    probs = forward(X, W, b)
    return np.argmax(probs, axis=1)


def accuracy(X, y, W, b):
    """Compute classification accuracy."""
    predictions = predict(X, W, b)
    return np.mean(predictions == y)


# =============================================================================
# STEP 6C: THE TRAINING LOOP
# =============================================================================

print("-" * 40)
print("TRAINING:")
print("-" * 40)
print()

# Hyperparameters
learning_rate = 1.0  # How big our update steps are
num_iterations = 200  # How many times to update

# Track history
train_losses = []
train_accuracies = []
test_accuracies = []

print(f"Learning rate: {learning_rate}")
print(f"Iterations: {num_iterations}")
print()
print(f"{'Iter':<6} {'Loss':<10} {'Train Acc':<12} {'Test Acc':<12}")
print("-" * 40)

for i in range(num_iterations):

    # ===== STEP 1: FORWARD PASS =====
    probs = forward(X_train, W, b)

    # ===== STEP 2: COMPUTE LOSS =====
    loss = compute_loss(probs, y_train)
    train_losses.append(loss)

    # ===== STEP 3: COMPUTE GRADIENTS (BACKWARD PASS) =====
    dW, db = compute_gradients(X_train, y_train, probs)

    # ===== STEP 4: UPDATE PARAMETERS =====
    # Gradient descent: move opposite to gradient
    W = W - learning_rate * dW
    b = b - learning_rate * db

    # Track accuracy
    train_acc = accuracy(X_train, y_train, W, b)
    test_acc = accuracy(X_test, y_test, W, b)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # Print progress
    if i % 20 == 0 or i == num_iterations - 1:
        print(f"{i:<6} {loss:<10.4f} {train_acc:<12.1%} {test_acc:<12.1%}")

print("-" * 40)
print()

# =============================================================================
# STEP 6D: Final results
# =============================================================================

print("-" * 40)
print("FINAL RESULTS:")
print("-" * 40)
print()

final_train_acc = accuracy(X_train, y_train, W, b)
final_test_acc = accuracy(X_test, y_test, W, b)

print(f"Final training accuracy: {final_train_acc:.1%}")
print(f"Final test accuracy:     {final_test_acc:.1%}")
print()

if final_test_acc >= 0.9:
    print("Excellent! The model generalizes well to unseen data.")
elif final_test_acc >= 0.7:
    print("Good! The model learned useful patterns.")
else:
    print("The model needs improvement.")
print()

# =============================================================================
# STEP 6E: What did the model learn?
# =============================================================================

print("-" * 40)
print("WHAT DID THE MODEL LEARN?")
print("-" * 40)
print()

feature_names = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']

print("Learned weights W (each column = template for a class):")
print()
print(f"{'Feature':<12} {'Setosa':<10} {'Versicolor':<12} {'Virginica':<10}")
print("-" * 44)
for i, name in enumerate(feature_names):
    print(f"{name:<12} {W[i,0]:<10.3f} {W[i,1]:<12.3f} {W[i,2]:<10.3f}")
print()

print("Interpretation:")
print("  - Large POSITIVE weight = feature suggests this class")
print("  - Large NEGATIVE weight = feature suggests NOT this class")
print()

# Find most important features for each class
for c in range(num_classes):
    class_name = target_names[c]
    weights = W[:, c]
    most_positive = feature_names[np.argmax(weights)]
    most_negative = feature_names[np.argmin(weights)]
    print(f"{class_name}:")
    print(f"  Most indicative:     {most_positive} (weight={weights.max():.2f})")
    print(f"  Least indicative:    {most_negative} (weight={weights.min():.2f})")
    print()

# =============================================================================
# STEP 6F: Visualize training
# =============================================================================

print("-" * 40)
print("GENERATING PLOTS...")
print("-" * 40)
print()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot 1: Loss curve
axes[0].plot(train_losses, 'b-', linewidth=2)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)

# Plot 2: Accuracy curves
axes[1].plot(train_accuracies, 'b-', label='Train', linewidth=2)
axes[1].plot(test_accuracies, 'r--', label='Test', linewidth=2)
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training vs Test Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1.05])

# Plot 3: Confusion matrix (final predictions)
from collections import Counter

predictions_test = predict(X_test, W, b)
confusion = np.zeros((num_classes, num_classes), dtype=int)
for true, pred in zip(y_test, predictions_test):
    confusion[true, pred] += 1

im = axes[2].imshow(confusion, cmap='Blues')
axes[2].set_xticks(range(num_classes))
axes[2].set_yticks(range(num_classes))
axes[2].set_xticklabels(target_names, rotation=45)
axes[2].set_yticklabels(target_names)
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('True')
axes[2].set_title('Confusion Matrix (Test Set)')

# Add numbers to confusion matrix
for i in range(num_classes):
    for j in range(num_classes):
        axes[2].text(j, i, confusion[i, j],
                     ha='center', va='center', fontsize=14,
                     color='white' if confusion[i, j] > confusion.max()/2 else 'black')

plt.tight_layout()
plt.savefig('/Users/madhavayyagari/ml-from-scratch/step6_training_results.png', dpi=150)
plt.show()

print("Plot saved to step6_training_results.png")
print()

# =============================================================================
# STEP 6G: Save trained model
# =============================================================================

np.savez('/Users/madhavayyagari/ml-from-scratch/model_trained.npz',
         W=W, b=b)
print("Trained model saved to model_trained.npz")
print()

# =============================================================================
# KEY CONCEPTS
# =============================================================================

print("=" * 60)
print("LESSON 1 COMPLETE - KEY CONCEPTS:")
print("=" * 60)
print("""
THE TRAINING LOOP (memorize this!):
1. FORWARD:  probs = softmax(X @ W + b)
2. LOSS:     L = -mean(log(probs[correct_class]))
3. BACKWARD: dW = X.T @ (probs - one_hot(y))
4. UPDATE:   W = W - learning_rate * dW

WHAT YOU LEARNED:
- A classifier maps inputs to class probabilities
- Weights and bias are learned through gradient descent
- Loss function measures how wrong we are
- Gradients tell us how to improve
- Training = repeating forward-backward-update

WHAT'S NEXT?
This linear classifier has a limitation: it can only create
LINEAR decision boundaries (straight lines/planes).

In Lesson 2, we'll add HIDDEN LAYERS and NONLINEARITY (ReLU)
to create a neural network that can learn curved boundaries.

Run: python lesson_2_neural_network.py (when ready)
""")
