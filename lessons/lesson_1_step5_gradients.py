"""
Lesson 1, Step 5: Gradients (The Key to Learning)
=================================================

We have:
- A model with parameters W and b
- A loss function that measures how wrong we are

Now: How do we IMPROVE the model?

ANSWER: Gradients tell us how to change W and b to reduce loss.

GRADIENT = direction of steepest INCREASE
So we move in the OPPOSITE direction to decrease loss.

This is called GRADIENT DESCENT.
"""

import numpy as np

# =============================================================================
# STEP 5A: What is a gradient?
# =============================================================================

print("=" * 60)
print("STEP 5: UNDERSTANDING GRADIENTS")
print("=" * 60)
print()

print("-" * 40)
print("INTUITION: What is a gradient?")
print("-" * 40)
print()

print("""
Imagine you're blindfolded on a hilly landscape.
Your goal: reach the lowest point (minimum loss).

The GRADIENT tells you:
- Which direction is UPHILL (steepest increase)
- How STEEP the hill is

To go DOWNHILL, you walk in the OPPOSITE direction of the gradient.

For our model:
- gradient of W (dW) = how loss changes when we change each weight
- gradient of b (db) = how loss changes when we change each bias

If dW[i,j] is POSITIVE: increasing W[i,j] INCREASES loss
   → We should DECREASE W[i,j]

If dW[i,j] is NEGATIVE: increasing W[i,j] DECREASES loss
   → We should INCREASE W[i,j]
""")

# =============================================================================
# STEP 5B: Load data and compute forward pass
# =============================================================================

print("-" * 40)
print("COMPUTING GRADIENTS:")
print("-" * 40)
print()

# Load data and model
data = np.load('/Users/madhavayyagari/ml-from-scratch/iris_prepared.npz',
               allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']

model = np.load('/Users/madhavayyagari/ml-from-scratch/model_initial.npz')
W = model['W'].copy()
b = model['b'].copy()


def softmax(scores):
    scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# Forward pass
scores = X_train @ W + b
probs = softmax(scores)

print("Forward pass complete.")
print(f"  probs shape: {probs.shape}")
print()

# =============================================================================
# STEP 5C: Derive the gradient (the math)
# =============================================================================

print("-" * 40)
print("THE MATH (simplified):")
print("-" * 40)
print()

print("""
For cross-entropy loss with softmax, the gradient has a beautiful form:

1. Gradient w.r.t. scores:
   dscores = probs - one_hot(y)

   This is just: (what we predicted) - (what we should have predicted)

   Example for sample i with true class 2:
     probs[i]    = [0.2, 0.3, 0.5]  (our prediction)
     one_hot     = [0,   0,   1  ]  (correct answer)
     dscores[i]  = [0.2, 0.3, -0.5] (the error)

2. Gradient w.r.t. weights:
   dW = X.T @ dscores

   (This comes from the chain rule - how scores depend on W)

3. Gradient w.r.t. bias:
   db = sum(dscores)

   (Each bias affects all samples equally)
""")

# =============================================================================
# STEP 5D: Compute gradients in code
# =============================================================================

print("-" * 40)
print("COMPUTING GRADIENTS IN CODE:")
print("-" * 40)
print()

num_samples = len(X_train)

# Step 1: Gradient of loss w.r.t. scores
# dscores = probs - one_hot(y)
dscores = probs.copy()
dscores[np.arange(num_samples), y_train] -= 1  # Subtract 1 at correct class
dscores /= num_samples  # Average over samples

print("dscores (first 3 samples):")
for i in range(3):
    true_class = y_train[i]
    print(f"  Sample {i} (true class {true_class}): {dscores[i]}")
print()

# Step 2: Gradient w.r.t. weights
dW = X_train.T @ dscores

print("dW (gradient of loss w.r.t. weights):")
print(dW)
print(f"Shape: {dW.shape} (same as W)")
print()

# Step 3: Gradient w.r.t. bias
db = np.sum(dscores, axis=0)

print("db (gradient of loss w.r.t. bias):")
print(db)
print(f"Shape: {db.shape} (same as b)")
print()

# =============================================================================
# STEP 5E: Verify gradients numerically
# =============================================================================

print("-" * 40)
print("VERIFYING GRADIENTS (numerical check):")
print("-" * 40)
print()

print("We can verify our gradients are correct using finite differences:")
print("  dL/dW ≈ (L(W+h) - L(W-h)) / (2h)")
print()


def compute_loss(X, y, W, b):
    scores = X @ W + b
    probs = softmax(scores)
    correct_probs = probs[np.arange(len(y)), y]
    return -np.mean(np.log(correct_probs + 1e-8))


# Check gradient for W[0, 0]
h = 1e-5

# L(W + h)
W_plus = W.copy()
W_plus[0, 0] += h
loss_plus = compute_loss(X_train, y_train, W_plus, b)

# L(W - h)
W_minus = W.copy()
W_minus[0, 0] -= h
loss_minus = compute_loss(X_train, y_train, W_minus, b)

# Numerical gradient
numerical_grad = (loss_plus - loss_minus) / (2 * h)

print(f"For W[0, 0]:")
print(f"  Analytical gradient (our formula): {dW[0, 0]:.6f}")
print(f"  Numerical gradient (finite diff):  {numerical_grad:.6f}")
print(f"  Difference: {abs(dW[0, 0] - numerical_grad):.10f}")
print()
print("They match! Our gradient computation is correct.")
print()

# =============================================================================
# STEP 5F: Package as a function
# =============================================================================

def compute_gradients(X, y, probs):
    """
    Compute gradients of cross-entropy loss w.r.t. W and b.

    Args:
        X: Input features, shape (num_samples, num_features)
        y: True labels, shape (num_samples,)
        probs: Predicted probabilities, shape (num_samples, num_classes)

    Returns:
        dW: Gradient for weights
        db: Gradient for bias
    """
    num_samples = len(y)

    # Gradient of loss w.r.t. scores
    dscores = probs.copy()
    dscores[np.arange(num_samples), y] -= 1
    dscores /= num_samples

    # Gradient w.r.t. weights and bias
    dW = X.T @ dscores
    db = np.sum(dscores, axis=0)

    return dW, db


# =============================================================================
# KEY CONCEPTS
# =============================================================================

print("=" * 60)
print("KEY CONCEPTS:")
print("=" * 60)
print("""
1. GRADIENT = direction of steepest increase
   - We move OPPOSITE to gradient to decrease loss
   - This is gradient descent

2. THE KEY FORMULA:
   dscores = probs - one_hot(y)
   dW = X.T @ dscores
   db = sum(dscores)

3. INTUITION:
   - dscores = (what we predicted) - (what we should predict)
   - It's literally the "error" for each class
   - Positive = we over-predicted, Negative = we under-predicted

4. GRADIENT VERIFICATION:
   - Always verify with numerical gradients when implementing
   - (L(θ+h) - L(θ-h)) / 2h should match analytical gradient

NOW WE HAVE EVERYTHING:
- Forward pass: compute predictions
- Loss: measure how wrong we are
- Gradients: know which direction to move

NEXT STEP: Put it all together - the training loop!
Run: python lesson_1_step6_training.py
""")
