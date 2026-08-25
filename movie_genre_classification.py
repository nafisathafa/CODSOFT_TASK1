import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD TRAINING DATASET
# ============================================================

print("===================================================")
print("        MOVIE GENRE CLASSIFICATION")
print("===================================================")

train_file = "train_data.txt"

data = pd.read_csv(
    train_file,
    sep=":::",
    engine="python",
    header=None,
    names=["ID", "Title", "Genre", "Description"]
)


# ============================================================
# 2. CLEAN DATA
# ============================================================

data = data.dropna(
    subset=["Description", "Genre"]
)

data["Description"] = (
    data["Description"]
    .astype(str)
    .str.strip()
)

data["Genre"] = (
    data["Genre"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 3. DISPLAY DATASET INFORMATION
# ============================================================

print("\nTotal movies:", len(data))

print(
    "\nNumber of genres:",
    data["Genre"].nunique()
)

print("\nGenres:")
print(
    data["Genre"].value_counts()
)

print("\nExample movie description:")
print(
    data["Description"].iloc[0]
)

print("\nActual genre:")
print(
    data["Genre"].iloc[0]
)


# ============================================================
# 4. CREATE INPUT (X) AND TARGET (y)
# ============================================================

X = data["Description"]
y = data["Genre"]


# ============================================================
# 5. SPLIT DATA INTO TRAINING AND VALIDATION SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_test)
)


# ============================================================
# 6. TF-IDF FEATURE EXTRACTION
# ============================================================

print(
    "\nConverting movie descriptions into TF-IDF features..."
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    "\nTF-IDF conversion completed!"
)

print(
    "Training TF-IDF shape:",
    X_train_tfidf.shape
)

print(
    "Validation TF-IDF shape:",
    X_test_tfidf.shape
)


# ============================================================
# 7. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

print(
    "\nTraining Logistic Regression model..."
)

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)

print(
    "Model training completed!"
)


# ============================================================
# 8. VALIDATION PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# 9. VALIDATION EVALUATION
# ============================================================

validation_accuracy = accuracy_score(
    y_test,
    y_pred
)

validation_macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

validation_weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n===================================================")
print("              VALIDATION RESULTS")
print("===================================================")

print(
    f"\nModel Accuracy: "
    f"{validation_accuracy:.4f}"
)

print(
    f"Model Accuracy (%): "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"\nMacro F1 Score: "
    f"{validation_macro_f1:.4f}"
)

print(
    f"Weighted F1 Score: "
    f"{validation_weighted_f1:.4f}"
)

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

print(
    "\nGenerating confusion matrix..."
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

fig, ax = plt.subplots(
    figsize=(14, 12)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d"
)

plt.title(
    "Movie Genre Classification - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Confusion matrix saved as: "
    "confusion_matrix.png"
)


# ============================================================
# 11. LOAD CODSOFT TEST DATASET
# ============================================================

test_data = pd.read_csv(
    "test_data.txt",
    sep=":::",
    engine="python",
    header=None,
    names=["ID", "Title", "Description"]
)

test_data = test_data.dropna(
    subset=["Description"]
)

test_data["Description"] = (
    test_data["Description"]
    .astype(str)
    .str.strip()
)

print("\n===================================================")
print("              CODSOFT TEST DATA")
print("===================================================")

print(
    "\nTest dataset loaded!"
)

print(
    "Number of test movies:",
    len(test_data)
)


# ============================================================
# 12. CONVERT TEST DATA TO TF-IDF
# ============================================================

test_tfidf = vectorizer.transform(
    test_data["Description"]
)

print(
    "Test TF-IDF shape:",
    test_tfidf.shape
)


# ============================================================
# 13. PREDICT TEST GENRES
# ============================================================

test_predictions = model.predict(
    test_tfidf
)

test_data["Predicted_Genre"] = (
    test_predictions
)


# ============================================================
# 14. DISPLAY SAMPLE PREDICTIONS
# ============================================================

print("\nSample Predictions:")

print(
    test_data[
        ["Title", "Predicted_Genre"]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 15. LOAD ACTUAL TEST ANSWERS
# ============================================================

solution_data = pd.read_csv(
    "test_data_solution.txt",
    sep=":::",
    engine="python",
    header=None,
    names=[
        "ID",
        "Title",
        "Genre",
        "Description"
    ]
)

solution_data = solution_data.dropna(
    subset=["Genre"]
)

solution_data["Genre"] = (
    solution_data["Genre"]
    .astype(str)
    .str.strip()
)

print(
    "\nTest solution loaded!"
)

print(
    "Number of actual test labels:",
    len(solution_data)
)


# ============================================================
# 16. CHECK DATASET LENGTHS
# ============================================================

if len(test_data) != len(solution_data):

    raise ValueError(
        "Test dataset and solution dataset "
        "have different numbers of rows."
    )


# ============================================================
# 17. COMPARE ACTUAL AND PREDICTED GENRES
# ============================================================

test_data["Actual_Genre"] = (
    solution_data["Genre"].values
)


# ============================================================
# 18. FINAL TEST EVALUATION
# ============================================================

test_accuracy = accuracy_score(
    test_data["Actual_Genre"],
    test_data["Predicted_Genre"]
)

test_macro_f1 = f1_score(
    test_data["Actual_Genre"],
    test_data["Predicted_Genre"],
    average="macro"
)

test_weighted_f1 = f1_score(
    test_data["Actual_Genre"],
    test_data["Predicted_Genre"],
    average="weighted"
)


print("\n===================================================")
print("                FINAL TEST RESULTS")
print("===================================================")

print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Final Test Accuracy (%): "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"\nFinal Test Macro F1: "
    f"{test_macro_f1:.4f}"
)

print(
    f"Final Test Weighted F1: "
    f"{test_weighted_f1:.4f}"
)


# ============================================================
# 19. ACTUAL VS PREDICTED
# ============================================================

comparison = test_data[
    [
        "Title",
        "Actual_Genre",
        "Predicted_Genre"
    ]
].copy()

comparison["Correct"] = (
    comparison["Actual_Genre"]
    ==
    comparison["Predicted_Genre"]
)

print(
    "\nSample Actual vs Predicted:"
)

print(
    comparison
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 20. COUNT CORRECT AND INCORRECT PREDICTIONS
# ============================================================

correct_predictions = (
    comparison["Correct"].sum()
)

incorrect_predictions = (
    (~comparison["Correct"]).sum()
)

print(
    "\nCorrect predictions:",
    correct_predictions
)

print(
    "Incorrect predictions:",
    incorrect_predictions
)


# ============================================================
# 21. SAVE PREDICTIONS
# ============================================================

comparison.to_csv(
    "movie_genre_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to:"
)

print(
    "movie_genre_predictions.csv"
)


# ============================================================
# 22. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "movie_genre_model.pkl"
)

print(
    "\nTrained model saved as:"
)

print(
    "movie_genre_model.pkl"
)


# ============================================================
# 23. SAVE TF-IDF VECTORIZER
# ============================================================

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)

print(
    "\nTF-IDF vectorizer saved as:"
)

print(
    "tfidf_vectorizer.pkl"
)


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("\n===================================================")
print("                 PROJECT SUMMARY")
print("===================================================")

print(
    f"\nNumber of training movies: "
    f"{len(X_train)}"
)

print(
    f"Number of validation movies: "
    f"{len(X_test)}"
)

print(
    f"Number of genres: "
    f"{y.nunique()}"
)

print(
    f"TF-IDF features: "
    f"{X_train_tfidf.shape[1]}"
)

print(
    "\nAlgorithm:"
)

print(
    "TF-IDF + Logistic Regression"
)

print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Final Test Macro F1: "
    f"{test_macro_f1:.4f}"
)

print(
    "\n==================================================="
)

print(
    "Movie Genre Classification completed successfully!"
)

print(
    "===================================================")