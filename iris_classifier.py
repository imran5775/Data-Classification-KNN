"""Iris flower classification with K-Nearest Neighbors.

DecodeLabs AI Internship, Project 2 | Author: Muhammad Imran

Pipeline (IPO):
    Input   -> load Iris data, explore it, scale the features
    Process -> shuffle + split 80/20, pick K, train KNN
    Output  -> accuracy, confusion matrix, precision / recall / F1
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
IMAGES_DIR = Path(__file__).parent / "images"


def explore(iris):
    """Print a quick summary so we understand the data before modelling."""
    X, y = iris.data, iris.target
    print("=== 1. DATASET OVERVIEW ===")
    print(f"Samples : {X.shape[0]}")
    print(f"Features: {X.shape[1]} -> {', '.join(iris.feature_names)}")
    print(f"Classes : {len(iris.target_names)} -> {', '.join(iris.target_names)}")
    counts = np.bincount(y)
    for name, count in zip(iris.target_names, counts):
        print(f"  {name:<11}: {count} samples")
    print(f"Missing values: {int(np.isnan(X).sum())}\n")


def find_best_k(X_train, y_train, max_k=25):
    """Try odd K values with 5-fold cross-validation and return the best one.

    A pipeline is used so the scaler is re-fitted inside each fold,
    which keeps validation data from leaking into training.
    """
    k_values = list(range(1, max_k + 1, 2))
    errors = []
    for k in k_values:
        model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
        errors.append(1 - accuracy)

    best_k = k_values[int(np.argmin(errors))]

    plt.figure(figsize=(7, 4))
    plt.plot(k_values, errors, marker="o")
    plt.axvline(best_k, color="red", linestyle="--", label=f"best K = {best_k}")
    plt.xlabel("K (number of neighbors)")
    plt.ylabel("Cross-validation error rate")
    plt.title("Choosing K for KNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "k_selection.png", dpi=150)
    return best_k


def main():
    parser = argparse.ArgumentParser(description="Iris classification with KNN")
    parser.add_argument("--k", type=int, default=None, help="use a fixed K instead of searching")
    parser.add_argument("--no-show", action="store_true", help="save plots without opening windows")
    args = parser.parse_args()

    IMAGES_DIR.mkdir(exist_ok=True)

    # ---------- INPUT ----------
    iris = load_iris()
    X, y = iris.data, iris.target
    explore(iris)

    # ---------- PROCESS ----------
    # Shuffled, stratified split keeps all 3 classes balanced in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=True, stratify=y, random_state=RANDOM_STATE
    )
    print("=== 2. TRAIN / TEST SPLIT ===")
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}\n")

    # Fit the scaler on training data only, then reuse it on the test data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("=== 3. CHOOSING K ===")
    k = args.k if args.k else find_best_k(X_train, y_train)
    print(f"Using K = {k}\n")

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    # ---------- OUTPUT ----------
    print("=== 4. EVALUATION ===")
    print(f"Accuracy      : {accuracy_score(y_test, predictions):.2%}")
    print(f"Macro F1 score: {f1_score(y_test, predictions, average='macro'):.3f}\n")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions), "\n")
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=iris.target_names, cmap="Blues"
    )
    plt.title(f"Confusion Matrix (KNN, K={k})")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=150)

    # Predict a brand-new flower the model has never seen.
    new_flower = [[5.1, 3.5, 1.4, 0.2]]
    guess = model.predict(scaler.transform(new_flower))[0]
    print(f"New flower {new_flower[0]} -> predicted species: {iris.target_names[guess]}")
    print(f"Plots saved in: {IMAGES_DIR}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

    