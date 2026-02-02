"""
Lesson 1, Step 1: Understanding Your Data
==========================================

Before ANY machine learning, you must understand your data.
We'll use the Iris dataset - the most famous beginner ML dataset.

THE IRIS DATASET:
- 150 flowers measured by botanists
- 3 species: Setosa, Versicolor, Virginica
- 4 measurements per flower:
    - Sepal length (cm)
    - Sepal width (cm)
    - Petal length (cm)
    - Petal width (cm)

OUR GOAL: Given the 4 measurements, predict the species.

This is CLASSIFICATION:
- Input: 4 numbers (the measurements)
- Output: 1 of 3 categories (the species)
"""

import numpy as np
from sklearn.datasets import load_iris

# =============================================================================
# STEP 1A: Load and explore the data
# =============================================================================

print("=" * 60)
print("STEP 1: UNDERSTANDING THE IRIS DATASET")
print("=" * 60)
print()

# Load the dataset (comes built into sklearn)
iris = load_iris()

# X contains the features (measurements)
# y contains the labels (species)
X = iris.data
y = iris.target

print("What do we have?")
print("-" * 40)
print(f"X shape: {X.shape}")
print(f"  → {X.shape[0]} flowers, {X.shape[1]} measurements each")
print()
print(f"y shape: {y.shape}")
print(f"  → {y.shape[0]} labels (one per flower)")
print()

# =============================================================================
# STEP 1B: Look at actual examples
# =============================================================================

print("Feature names (what we measure):")
for i, name in enumerate(iris.feature_names):
    print(f"  {i}: {name}")
print()

print("Class names (what we predict):")
for i, name in enumerate(iris.target_names):
    print(f"  {i}: {name}")
print()

print("-" * 40)
print("Let's look at 5 actual flowers:")
print("-" * 40)
print()

for i in range(5):
    species = iris.target_names[y[i]]
    print(f"Flower #{i+1}: {species}")
    print(f"  Sepal: {X[i, 0]:.1f}cm x {X[i, 1]:.1f}cm")
    print(f"  Petal: {X[i, 2]:.1f}cm x {X[i, 3]:.1f}cm")
    print()

# =============================================================================
# STEP 1C: Basic statistics - what patterns exist?
# =============================================================================

print("-" * 40)
print("Average measurements by species:")
print("-" * 40)
print()

for species_id in range(3):
    species_name = iris.target_names[species_id]
    species_data = X[y == species_id]  # Filter to just this species

    print(f"{species_name}:")
    print(f"  Avg sepal length: {species_data[:, 0].mean():.2f} cm")
    print(f"  Avg sepal width:  {species_data[:, 1].mean():.2f} cm")
    print(f"  Avg petal length: {species_data[:, 2].mean():.2f} cm")
    print(f"  Avg petal width:  {species_data[:, 3].mean():.2f} cm")
    print()

# =============================================================================
# STEP 1D: Visualize the data
# =============================================================================

import matplotlib.pyplot as plt

print("Generating visualization...")
print()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot each pair of features
feature_pairs = [(0, 1), (2, 3), (0, 2), (1, 3)]
titles = [
    'Sepal: Length vs Width',
    'Petal: Length vs Width',
    'Sepal Length vs Petal Length',
    'Sepal Width vs Petal Width'
]

colors = ['purple', 'green', 'orange']

for ax, (f1, f2), title in zip(axes.flat, feature_pairs, titles):
    for species_id in range(3):
        mask = y == species_id
        ax.scatter(X[mask, f1], X[mask, f2],
                   c=colors[species_id],
                   label=iris.target_names[species_id],
                   alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel(iris.feature_names[f1])
    ax.set_ylabel(iris.feature_names[f2])
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/madhavayyagari/ml-from-scratch/step1_data_exploration.png', dpi=150)
plt.show()

print("Plot saved to step1_data_exploration.png")
print()

# =============================================================================
# KEY OBSERVATIONS
# =============================================================================

print("=" * 60)
print("KEY OBSERVATIONS (look at the plots!):")
print("=" * 60)
print("""
1. SETOSA IS EASY TO SEPARATE
   - Look at 'Petal: Length vs Width' (top right)
   - Setosa (purple) is completely separate from the others
   - A simple rule could work: "if petal length < 2.5, it's Setosa"

2. VERSICOLOR vs VIRGINICA IS HARDER
   - They overlap in most plots
   - No single measurement perfectly separates them
   - We need to use MULTIPLE features together

3. SOME FEATURES ARE MORE USEFUL THAN OTHERS
   - Petal measurements separate species better than sepal
   - This is feature importance - some inputs matter more

4. THE DATA IS "MOSTLY" LINEARLY SEPARABLE
   - You could draw lines to mostly separate the classes
   - This is good news for our linear classifier!

NEXT STEP: We'll prepare this data for training.
Run: python lesson_1_step2_preparation.py
""")
