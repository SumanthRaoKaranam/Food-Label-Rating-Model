# Food Label Rating Model

A classic machine learning project that rates packaged food from **0 to 5** using the information printed on its label: the nutrition table and the ingredient list.

No LLMs, no RAG, no agents. Just feature engineering and XGBoost.

Allowed ratings: `0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5`

---

## Why this project

Food labels are hard to read quickly. The idea is simple: a customer scans the label on a packet, and the system tells them whether the product is a good or bad choice on an easy 0 to 5 scale.

This repository covers the **machine learning part**: nutrition numbers and ingredients go in, a rating comes out. Reading text from a photo (OCR) is a separate step and is listed under future work.

---

## How it works

1. **Load data** from a Kaggle food dataset (a single CSV, about 6,500 products used).
2. **Clean it**: keep products with a full nutrition table and ingredient list, and remove impossible values (for example 900 g of sugar per 100 g).
3. **Build the 0 to 5 rating** (the dataset has no rating, so it is created, see below).
4. **Create features** that exist on a real label.
5. **Train an XGBoost regressor**, tuned with randomized search and cross-validation.
6. **Snap the prediction** to the nearest allowed rating.
7. **Evaluate** with R2 score, error measures and charts, then **save** the model with joblib.

---

## Dataset

[Global Food Nutrition Database (10k Products)](https://www.kaggle.com/datasets/kanchana1990/global-food-nutrition-database10k-products) on Kaggle. It includes Nutri-Score grades, NOVA groups, nutrition values per 100 g, and ingredient text.

The notebook downloads it automatically with `kagglehub`. If that fails, it asks you to upload the CSV by hand.

The dataset is mostly European, so many ingredient lists are in French, Spanish and other languages. The model treats those words as ordinary tokens.

---

## How the rating is built

The dataset has no 0 to 5 rating, so the target is created from three signals:

| Signal | Effect on rating |
|---|---|
| Nutri-Score letter | Base score: A = 5, B = 4, C = 3, D = 2, E = 1 |
| NOVA group (how processed) | NOVA 3 subtracts 0.5, NOVA 4 subtracts 1.0 |
| Additive count | 4 or more additives subtracts 0.5 |

The result is limited to 0 to 5 and rounded to the nearest allowed value.

**Important:** NOVA and the official additive count are used only to build the target. They are **not** given to the model. The model sees only what is printed on a label.

---

## Features used by the model

- Energy (kcal), fat, saturated fat, sugars, fiber, protein, salt (all per 100 g)
- Number of ingredients
- Number of E-number additives found in the ingredient text (for example E330)
- TF-IDF of the ingredient text (top 150 words)

---

## Model

- **XGBoost regressor** inside a scikit-learn pipeline
- Hyperparameters tuned with `RandomizedSearchCV` (3-fold cross-validation, scored on mean absolute error)
- Output rounded to the nearest allowed rating

Ratings are ordered, so regression fits better than plain classification: predicting 4 when the truth is 4.5 is a smaller mistake than predicting 1.

---

## Results

R2 score on the held-out test set (20% of the data):

| Measure | R2 score |
|---|---|
| Test, before rounding | 0.880 |
| Test, after rounding to allowed ratings | 0.873 |
| Train | 0.975 |

The model explains about 88% of the variation in the ratings on products it has never seen. The notebook also produces a true-vs-predicted heatmap and a feature importance chart.

---

## How to run

1. Open `food_label_rating.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run the cells from top to bottom.
3. The trained model is saved as `food_rating_model.joblib`.

To run locally instead:

```bash
pip install -r requirements.txt
jupyter notebook food_label_rating.ipynb
```

---

## Using the saved model

```python
import joblib, re
import pandas as pd

saved = joblib.load("food_rating_model.joblib")
model, allowed = saved["pipeline"], saved["allowed"]

def snap(v):
    return allowed[abs(allowed - v).argmin()]

def rate_product(energy_kcal, fat, sat_fat, sugars, fiber, protein, salt, ingredients_text):
    ing = ingredients_text.lower()
    row = pd.DataFrame([{
        "energy_kcal": energy_kcal, "fat": fat, "sat_fat": sat_fat, "sugars": sugars,
        "fiber": fiber, "protein": protein, "salt": salt,
        "ingredient_count": len(re.split(r"[,;]", ing)),
        "additive_count": len(re.findall(r"\be\s?\d{3,4}[a-i]?\b", ing)),
        "ingredients": ing}])
    return float(snap(model.predict(row)[0]))

# all numbers are per 100 g; salt is grams of salt (not sodium)
print(rate_product(520, 28, 11, 45, 2, 5, 0.4,
                   "sugar, palm oil, wheat flour, cocoa, emulsifier (E322), flavouring"))
```

---

## Project structure

```
.
├── food_label_rating.ipynb    # full notebook (data, training, evaluation)
├── food_rating_model.joblib   # saved model (created after running)
├── requirements.txt
└── README.md
```

---

## Limitations

- The 0 to 5 rating comes from a formula built on Nutri-Score, NOVA and additives. It is **not** a human or medical opinion, so the model is learning that formula from label data.
- The model scores higher on training data than on test data (0.975 vs 0.880), so it fits the training set more closely than new products. More data or simpler settings could narrow the gap.
- The dataset is mostly European, so results on other markets may be weaker.
- Nutrition values must be per 100 g. Salt must be grams of salt, not sodium.
- This is a learning project and not nutritional or medical advice.

---

## Future work

- Add OCR (for example Tesseract) to read the nutrition table and ingredients from a photo of the label.
- Add a simple web or mobile front end.
- Test on products from more countries and languages.
- Compare the model against other approaches.

---

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, joblib, Google Colab
