"""
Lesson 1, Step 4: The Loss Function
====================================

The loss function measures HOW WRONG our predictions are.
It's a single number that we want to minimize.

We use CROSS-ENTROPY LOSS:
    loss = -log(probability of correct class)

INTUITION:
- If we predict 90% for the correct class → loss is low (good!)
- If we predict 10% for the correct class → loss is high (bad!)
- If we predict 1% for the correct class → loss is VERY high (very bad!)

The -log() function penalizes confident wrong predictions harshly.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# STEP 4A: Understanding -log(p) visually
# =============================================================================

print("=" * 60)
print("STEP 4: UNDERSTANDING THE LOSS FUNCTION")
print("=" * 60)
print()

print("-" * 40)
print("WHY -log(probability)?")
print("-" * 40)
print()

# Let's visualize -log(p)
p = np.linspace(0.01, 1.0, 100)  # Probabilities from 1% to 100%
loss = -np.log(p)

print("If our predicted probability for the CORRECT class is:")
print()
examples = [0.95, 0.80, 0.50, 0.20, 0.05, 0.01]
for prob in examples:
    l = -np.log(prob)
    print(f"  {prob*100:5.1f}%  →  loss = {l:.3f}")
print()
print("Notice: loss grows RAPIDLY as probability drops!")
print("This heavily penalizes confident wrong predictions.")
print()

# Plot it
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(p, loss, 'b-', linewidth=2)
plt.xlabel('Predicted probability for correct class')
plt.ylabel('Loss: -log(p)')
plt.title('Cross-Entropy Loss Function')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)

# Annotate some points
for prob in [0.9, 0.5, 0.1]:
    l = -np.log(prob)
    plt.plot(prob, l, 'ro', markersize=8)
    plt.annotate(f'p={prob}, loss={l:.2f}',
                 xy=(prob, l), xytext=(prob+0.1, l+0.5),
                 fontsize=9)

plt.subplot(1, 2, 2)
plt.plot(p, loss, 'b-', linewidth=2)
plt.xlabel('Predicted probability for correct class')
plt.ylabel('Loss: -log(p)')
plt.title('Same plot, log scale')
plt.yscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/madhavayyagari/ml-from-scratch/step4_loss_function.png', dpi=150)
plt.show()
print("Plot saved to step4_loss_function.png")
print()

# =============================================================================
# STEP 4B: Load our model and data
# =============================================================================

print("-" * 40)
print("COMPUTING LOSS ON OUR DATA:")
print("-" * 40)
print()

# Load data and model
data = np.load('/Users/madhavayyagari/ml-from-scratch/iris_prepared.npz',
               allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']

model = np.load('/Users/madhavayyagari/ml-from-scratch/model_initial.npz')
W = model['W']
b = model['b']

print(f"Loaded {len(X_train)} training samples")
print()

# =============================================================================
# STEP 4C: Compute predictions and loss step by step
# =============================================================================

# Forward pass
def softmax(scores):
    scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

scores = X_train @ W + b
probs = softmax(scores)

print("Step-by-step loss calculation:")
print()

# Show first 5 examples
print("First 5 training examples:")
print("-" * 60)
print(f"{'Sample':<8} {'True':<6} {'P(true)':<10} {'Loss':<10}")
print("-" * 60)

for i in range(5):
    true_class = y_train[i]
    prob_true = probs[i, true_class]
    loss_i = -np.log(prob_true + 1e-8)  # +epsilon to avoid log(0)
    print(f"{i:<8} {true_class:<6} {prob_true:<10.4f} {loss_i:<10.4f}")
print()

# =============================================================================
# STEP 4D: Average loss over all samples
# =============================================================================

print("-" * 40)
print("TOTAL LOSS:")
print("-" * 40)
print()


def compute_loss(probs, y):
    """
    Cross-entropy loss.

    Args:
        probs: Predicted probabilities, shape (num_samples, num_classes)
        y: True labels, shape (num_samples,)

    Returns:
        loss: Average cross-entropy loss (single number)
    """
    num_samples = len(y)

    # Get probability of the true class for each sample
    # probs[i, y[i]] gives P(correct class) for sample i
    correct_probs = probs[np.arange(num_samples), y]

    # Cross-entropy: -log(P(correct class))
    # Average over all samples
    loss = -np.mean(np.log(correct_probs + 1e-8))

    return loss


loss = compute_loss(probs, y_train)
print(f"Average loss with random weights: {loss:.4f}")
print()

# What's a good baseline?
random_guess_loss = -np.log(1/3)  # If we guess randomly (33% each)
print(f"Loss if guessing randomly (33%): {random_guess_loss:.4f}")
print()

if loss < random_guess_loss:
    print("Our random model is slightly better than random guessing.")
else:
    print("Our random model is about as good as random guessing.")
print()

# =============================================================================
# STEP 4E: What loss should we aim for?
# =============================================================================

print("-" * 40)
print("LOSS TARGETS:")
print("-" * 40)
print()

print("Perfect predictions (100% confident and correct):")
print(f"  loss = -log(1.0) = {-np.log(1.0):.4f}")
print()

print("Very good predictions (90% confident and correct):")
print(f"  loss = -log(0.9) = {-np.log(0.9):.4f}")
print()

print("Our current loss: {:.4f}".format(loss))
print()
print("GOAL: Drive loss as low as possible during training!")
print()

# =============================================================================
# KEY CONCEPTS
# =============================================================================

print("=" * 60)
print("KEY CONCEPTS:")
print("=" * 60)
print("""
1. LOSS FUNCTION = measure of how wrong we are
   - Single number summarizing model performance
   - Lower is better

2. CROSS-ENTROPY LOSS: -log(P(correct class))
   - High confidence, correct → low loss
   - High confidence, wrong → VERY high loss
   - This is what we minimize during training

3. WHY CROSS-ENTROPY?
   - Comes from information theory
   - Has nice mathematical properties (smooth gradients)
   - Standard for classification tasks

4. AVERAGE LOSS
   - We compute loss for each sample, then average
   - This gives us one number to optimize

CURRENT STATE:
- Random weights → loss ≈ 1.1 (random guessing)
- Goal: loss → 0 (perfect predictions)

QUESTION: How do we REDUCE the loss?
ANSWER: Gradients! They tell us which direction to move.

NEXT STEP: Compute gradients (backpropagation).
Run: python lesson_1_step5_gradients.py
""")
