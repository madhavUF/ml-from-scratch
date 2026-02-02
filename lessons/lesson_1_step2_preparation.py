"""
Lesson 1, Step 2: Preparing Data for ML
========================================

Before training, we need to:
1. Normalize the features (make them similar scale)
2. Split into training and test sets (to check if we're really learning)

WHY NORMALIZE?
- Sepal length ranges from 4.3 to 7.9 cm
- Petal width ranges from 0.1 to 2.5 cm
- Without normalization, sepal length would dominate just because
  its numbers are bigger, not because it's more important.

WHY SPLIT?
- If we test on the same data we trained on, we're just checking
  if the model memorized the answers (overfitting)
- Test set = data the model has NEVER seen during training
- This tells us if the model actually learned general patterns
"""

import numpy as np
from sklearn.datasets import load_iris

# =============================================================================
# STEP 2A: Load the data (same as step 1)
# =============================================================================

print("=" * 60)
print("STEP 2: PREPARING DATA FOR MACHINE LEARNING")
print("=" * 60)
print()

iris = load_iris()
X = iris.data.copy()  # Copy so we don't modify original
y = iris.target.copy()

print(f"Original data: {X.shape[0]} samples, {X.shape[1]} features")
print()

# =============================================================================
# STEP 2B: Look at the scale problem
# =============================================================================

print("-" * 40)
print("THE SCALE PROBLEM:")
print("-" * 40)
print()

print("Feature ranges BEFORE normalization:")
for i, name in enumerate(iris.feature_names):
    min_val = X[:, i].min()
    max_val = X[:, i].max()
    mean_val = X[:, i].mean()
    print(f"  {name}:")
    print(f"    min={min_val:.1f}, max={max_val:.1f}, mean={mean_val:.2f}")
print()

print("Problem: Sepal length (4-8) has bigger numbers than petal width (0.1-2.5)")
print("The model might think sepal length is 'more important' just because")
print("the numbers are bigger. That's not fair!")
print()

# =============================================================================
# STEP 2C: Normalize the features
# =============================================================================

print("-" * 40)
print("NORMALIZING (Standardization):")
print("-" * 40)
print()

print("Formula: X_normalized = (X - mean) / std")
print()
print("This transforms each feature to have:")
print("  - Mean = 0 (centered at zero)")
print("  - Standard deviation = 1 (similar spread)")
print()

# Calculate mean and std for each feature
means = X.mean(axis=0)  # Mean of each column
stds = X.std(axis=0)    # Std of each column

print("Computed statistics:")
for i, name in enumerate(iris.feature_names):
    print(f"  {name}: mean={means[i]:.2f}, std={stds[i]:.2f}")
print()

# Normalize
X_normalized = (X - means) / stds

print("Feature ranges AFTER normalization:")
for i, name in enumerate(iris.feature_names):
    min_val = X_normalized[:, i].min()
    max_val = X_normalized[:, i].max()
    mean_val = X_normalized[:, i].mean()
    print(f"  {name}:")
    print(f"    min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}")
print()

print("Now all features are on similar scales!")
print()

# =============================================================================
# STEP 2D: Split into training and test sets
# =============================================================================

print("-" * 40)
print("TRAIN/TEST SPLIT:")
print("-" * 40)
print()

print("We'll use 80% for training, 20% for testing.")
print()

# Shuffle the data first (so we don't get all one species at the end)
np.random.seed(42)  # For reproducibility
shuffle_idx = np.random.permutation(len(X_normalized))
X_shuffled = X_normalized[shuffle_idx]
y_shuffled = y[shuffle_idx]

# Split
split_point = int(0.8 * len(X_shuffled))

X_train = X_shuffled[:split_point]
y_train = y_shuffled[:split_point]

X_test = X_shuffled[split_point:]
y_test = y_shuffled[split_point:]

print(f"Training set: {len(X_train)} samples")
print(f"Test set:     {len(X_test)} samples")
print()

# Check class distribution
print("Class distribution in training set:")
for i, name in enumerate(iris.target_names):
    count = np.sum(y_train == i)
    print(f"  {name}: {count} samples ({count/len(y_train)*100:.0f}%)")
print()

print("Class distribution in test set:")
for i, name in enumerate(iris.target_names):
    count = np.sum(y_test == i)
    print(f"  {name}: {count} samples ({count/len(y_test)*100:.0f}%)")
print()

# =============================================================================
# STEP 2E: Save prepared data for next step
# =============================================================================

print("-" * 40)
print("SAVING PREPARED DATA:")
print("-" * 40)
print()

# Save to .npz file (NumPy's compressed format)
np.savez('/Users/madhavayyagari/ml-from-scratch/iris_prepared.npz',
         X_train=X_train,
         y_train=y_train,
         X_test=X_test,
         y_test=y_test,
         feature_names=iris.feature_names,
         target_names=iris.target_names,
         means=means,
         stds=stds)

print("Saved to iris_prepared.npz")
print()
print("This file contains:")
print("  - X_train, y_train: Training data")
print("  - X_test, y_test: Test data")
print("  - means, stds: For normalizing new data later")
print()

# =============================================================================
# KEY CONCEPTS
# =============================================================================

print("=" * 60)
print("KEY CONCEPTS:")
print("=" * 60)
print("""
1. NORMALIZATION
   - Makes all features comparable
   - Helps the model learn faster and better
   - Formula: (value - mean) / std

2. TRAIN/TEST SPLIT
   - Training data: model learns from this
   - Test data: we evaluate on this (model never sees during training)
   - Prevents "cheating" (memorization vs learning)

3. SHUFFLING
   - Randomize order before splitting
   - Ensures both sets have mix of all classes

4. REPRODUCIBILITY
   - np.random.seed(42) makes random operations repeatable
   - Important for debugging and sharing results

WHAT WE HAVE NOW:
- 120 training samples (to learn from)
- 30 test samples (to evaluate on)
- All features normalized to similar scale

NEXT STEP: Build the linear classifier model.
Run: python lesson_1_step3_model.py
""")
