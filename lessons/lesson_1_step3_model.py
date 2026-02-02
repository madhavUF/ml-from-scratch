"""
Lesson 1, Step 3: The Linear Classifier Model
==============================================

Now we build the actual model. A linear classifier is simple:

    scores = X @ W + b

Where:
- X = input features (flower measurements)
- W = weights (what the model learns)
- b = bias (what the model learns)
- scores = one number per class (higher = more likely)

INTUITION:
Think of each class having a "template" (a column in W).
The model computes how similar the input is to each template.
The most similar template wins.
"""

import numpy as np

# =============================================================================
# STEP 3A: Load prepared data
# =============================================================================

print("=" * 60)
print("STEP 3: BUILDING THE LINEAR CLASSIFIER MODEL")
print("=" * 60)
print()

data = np.load('/Users/madhavayyagari/ml-from-scratch/iris_prepared.npz',
               allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

num_features = X_train.shape[1]  # 4
num_classes = len(np.unique(y_train))  # 3

print(f"Loaded data: {len(X_train)} training, {len(X_test)} test samples")
print(f"Features: {num_features}, Classes: {num_classes}")
print()

# =============================================================================
# STEP 3B: Initialize weights and bias
# =============================================================================

print("-" * 40)
print("INITIALIZING PARAMETERS:")
print("-" * 40)
print()

print("W (weights) shape: (num_features, num_classes) = (4, 3)")
print("  → Each column is a 'template' for one class")
print("  → We have 4 weights for each of 3 classes = 12 weights total")
print()

print("b (bias) shape: (num_classes,) = (3,)")
print("  → One bias per class")
print("  → Lets the model prefer certain classes")
print()

# Initialize randomly (small values)
np.random.seed(42)
W = np.random.randn(num_features, num_classes) * 0.01
b = np.zeros(num_classes)

print("Initial W (random, small values):")
print(W)
print()
print("Initial b (zeros):")
print(b)
print()

# =============================================================================
# STEP 3C: Forward pass - making predictions
# =============================================================================

print("-" * 40)
print("FORWARD PASS (Making Predictions):")
print("-" * 40)
print()

print("Step 1: Compute raw scores")
print("  scores = X @ W + b")
print()

# Let's trace through ONE example
example_idx = 0
x_example = X_train[example_idx:example_idx+1]  # Shape (1, 4)
y_example = y_train[example_idx]

print(f"Example flower (class {y_example}):")
print(f"  Features (normalized): {x_example[0]}")
print()

# Compute scores
scores = x_example @ W + b
print(f"Raw scores: {scores[0]}")
print(f"  → Score for Setosa:     {scores[0, 0]:.4f}")
print(f"  → Score for Versicolor: {scores[0, 1]:.4f}")
print(f"  → Score for Virginica:  {scores[0, 2]:.4f}")
print()

# =============================================================================
# STEP 3D: Softmax - converting scores to probabilities
# =============================================================================

print("-" * 40)
print("SOFTMAX (Scores → Probabilities):")
print("-" * 40)
print()

print("Raw scores can be any number (positive or negative).")
print("We want probabilities that:")
print("  1. Are all positive")
print("  2. Sum to 1")
print()
print("Softmax formula: prob[i] = exp(score[i]) / sum(exp(scores))")
print()

# Softmax (with numerical stability trick)
scores_shifted = scores - np.max(scores)  # Subtract max for stability
exp_scores = np.exp(scores_shifted)
probs = exp_scores / np.sum(exp_scores)

print("After softmax:")
print(f"  P(Setosa)     = {probs[0, 0]:.4f} = {probs[0, 0]*100:.1f}%")
print(f"  P(Versicolor) = {probs[0, 1]:.4f} = {probs[0, 1]*100:.1f}%")
print(f"  P(Virginica)  = {probs[0, 2]:.4f} = {probs[0, 2]*100:.1f}%")
print(f"  Sum = {probs.sum():.4f} (always 1.0)")
print()

predicted_class = np.argmax(probs)
print(f"Predicted class: {predicted_class}")
print(f"Actual class: {y_example}")
print()

# =============================================================================
# STEP 3E: Let's package this as functions
# =============================================================================

print("-" * 40)
print("PACKAGING AS REUSABLE FUNCTIONS:")
print("-" * 40)
print()


def softmax(scores):
    """Convert raw scores to probabilities."""
    scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def forward(X, W, b):
    """
    Forward pass: input → scores → probabilities.

    Args:
        X: Input features, shape (num_samples, num_features)
        W: Weights, shape (num_features, num_classes)
        b: Bias, shape (num_classes,)

    Returns:
        probs: Predicted probabilities, shape (num_samples, num_classes)
    """
    scores = X @ W + b
    probs = softmax(scores)
    return probs


def predict(X, W, b):
    """Return predicted class labels."""
    probs = forward(X, W, b)
    return np.argmax(probs, axis=1)


# Test on all training data
probs_all = forward(X_train, W, b)
predictions = predict(X_train, W, b)
accuracy = np.mean(predictions == y_train)

print("Testing forward pass on all training data...")
print(f"Predictions shape: {predictions.shape}")
print(f"Accuracy with random weights: {accuracy:.1%}")
print()
print("(This is basically random guessing - 33% for 3 classes)")
print("The model hasn't learned anything yet!")
print()

# =============================================================================
# STEP 3F: Save model for next step
# =============================================================================

np.savez('/Users/madhavayyagari/ml-from-scratch/model_initial.npz',
         W=W, b=b)
print("Saved initial model to model_initial.npz")
print()

# =============================================================================
# KEY CONCEPTS
# =============================================================================

print("=" * 60)
print("KEY CONCEPTS:")
print("=" * 60)
print("""
1. LINEAR TRANSFORMATION: scores = X @ W + b
   - Matrix multiplication + bias addition
   - This is ALL a linear classifier does

2. WEIGHTS (W)
   - Shape: (num_features, num_classes)
   - Each column = "template" for a class
   - These are what the model LEARNS

3. BIAS (b)
   - Shape: (num_classes,)
   - Lets the model favor certain classes
   - Also learned during training

4. SOFTMAX
   - Converts arbitrary scores to probabilities
   - exp() makes all values positive
   - Dividing by sum makes them sum to 1

5. PREDICTION
   - Pick the class with highest probability
   - argmax(probs)

CURRENT STATE:
- Model has random weights → random predictions (~33% accuracy)
- Next: We need to LEARN better weights!

NEXT STEP: Define the loss function (how wrong are we?).
Run: python lesson_1_step4_loss.py
""")
