# IMDB Movie Review Sentiment Prediction

A text classification project that predicts whether an IMDB movie review is **positive** or **negative** using TF-IDF features and Naive Bayes.

## Overview

This project trains a simple sentiment analysis pipeline on labeled IMDB movie reviews, classifying each review as positive (`1`) or negative (`0`) sentiment.

## Dataset

- **File:** `imdb_labelled.csv`
- **Format:** Tab-separated, no header, two columns: `review` (text) and `sentiment` (`0` = negative, `1` = positive)
- Rows with missing sentiment labels are dropped; sentiment is cast to integer.

> Note: the dataset is loaded from `/content/imdb_labelled.csv` (a Google Colab path). Update this path if running locally.

## Approach

1. **Features:** `TfidfVectorizer` (English stop words removed)
2. **Model:** Multinomial Naive Bayes, wrapped together with the vectorizer in a single `sklearn` `Pipeline`
3. **Split:** 80% train / 20% test (`random_state=42`)

## Results (Test Set)

| Metric | Score |
|---|---|
| Accuracy | 76.74% |
| Precision (positive) | 0.76 |
| Recall (positive) | 0.90 |
| Precision (negative) | 0.79 |
| Recall (negative) | 0.56 |
| F1 Score (weighted avg) | 0.76 |

The model is better at catching positive reviews (high recall) than negative ones — negative reviews are more often misclassified as positive. This is a small dataset (86 test samples), so results should be read as directional rather than definitive.

## Requirements

```
pandas
numpy
scikit-learn
```

Install with:
```bash
pip install pandas numpy scikit-learn
```

## Usage

1. Place `imdb_labelled.csv` in your working directory (update the file path in the notebook if not using Colab).
2. Run `IMDB_text_based_prediction.ipynb` top to bottom.
3. To classify a new review:

```python
review = ["This movie was fantastic, I loved it!"]
prediction = model.predict(review)
print("Prediction:", prediction[0])  # 1 = positive, 0 = negative
```

## Project Structure

```
.
├── IMDB_text_based_prediction.ipynb   # Preprocessing, training, evaluation
├── imdb_labelled.csv                   # Dataset (add your own)
└── README.md
```

## Future Improvements

- Try TF-IDF with n-grams (bigrams/trigrams) to capture more context
- Compare against Logistic Regression, SVM, or Random Forest
- Use a larger, more balanced dataset (this one has only ~430 labeled rows)
- Try word embeddings (Word2Vec, GloVe) or transformer-based models (BERT) for richer text representations
