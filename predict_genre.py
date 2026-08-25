import joblib


# ============================================================
# LOAD TRAINED MODEL AND TF-IDF VECTORIZER
# ============================================================

print("Loading trained model...")

model = joblib.load("movie_genre_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Model loaded successfully!")


# ============================================================
# MOVIE GENRE PREDICTION
# ============================================================

print("\n===================================================")
print("             MOVIE GENRE PREDICTOR")
print("===================================================")

print("\nEnter a movie plot summary.")
print("Type 'exit' to stop the program.")


while True:

    print("\n-----------------------------------")

    plot = input("Movie plot: ").strip()

    # Exit program
    if plot.lower() == "exit":
        print("\nPrediction program finished.")
        break

    # Check empty input
    if not plot:
        print("Please enter a movie plot.")
        continue

    # ========================================================
    # CONVERT PLOT INTO TF-IDF FEATURES
    # ========================================================

    plot_tfidf = vectorizer.transform([plot])

    # ========================================================
    # PREDICT GENRE
    # ========================================================

    prediction = model.predict(plot_tfidf)

    predicted_genre = prediction[0]

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n-----------------------------------")
    print("Predicted Movie Genre:", predicted_genre)
    print("-----------------------------------")