# Credit Card Fraud Detection — Walkthrough

## What Was Built

A **Flask web application** that trains 4 machine-learning models on the `creditcard 2.csv` dataset and presents an interactive comparison dashboard.

### Files Created

| File | Purpose |
|---|---|
| [app.py](file:///Users/mayankrai/Documents/GitHub/AIML-Projects/CreditCardFraudDetection/app.py) | Flask backend — data pipeline, model training, REST APIs |
| [templates/index.html](file:///Users/mayankrai/Documents/GitHub/AIML-Projects/CreditCardFraudDetection/templates/index.html) | Single-page dashboard UI |
| [static/style.css](file:///Users/mayankrai/Documents/GitHub/AIML-Projects/CreditCardFraudDetection/static/style.css) | Dark-mode design system |
| [static/app.js](file:///Users/mayankrai/Documents/GitHub/AIML-Projects/CreditCardFraudDetection/static/app.js) | Frontend logic |
| [requirements.txt](file:///Users/mayankrai/Documents/GitHub/AIML-Projects/CreditCardFraudDetection/requirements.txt) | Python dependencies |

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** 🏆 | 94.92% | 97.83% | 91.84% | **94.74%** | 97.92% |
| Gradient Boosting | 93.91% | 96.74% | 90.82% | 93.68% | 98.64% |
| KNN | 92.89% | 94.68% | 90.82% | 92.71% | 96.99% |
| SVM | 92.39% | 95.60% | 88.78% | 92.06% | 98.50% |

**Best model: Random Forest** (highest F1 score at 94.74%)

---

## Verified UI

### Dashboard with 4 model comparison cards
![Dashboard — all 4 model cards with metrics and the best-model banner](/Users/mayankrai/.gemini/antigravity/brain/e65df0a3-6479-4fe1-9fc5-39dddb1f41be/initial_dashboard_view_1771650704701.png)

### Model switching — KNN selected
![KNN selected — confusion matrix and metrics update](/Users/mayankrai/.gemini/antigravity/brain/e65df0a3-6479-4fe1-9fc5-39dddb1f41be/knn_selected_view_1771650719147.png)

### Prediction form with fraud result
![Prediction result showing Fraud with 80% confidence](/Users/mayankrai/.gemini/antigravity/brain/e65df0a3-6479-4fe1-9fc5-39dddb1f41be/prediction_result_view_1771650837640.png)

### Full demo recording
![Full app walkthrough recording](/Users/mayankrai/.gemini/antigravity/brain/e65df0a3-6479-4fe1-9fc5-39dddb1f41be/app_dashboard_verify_1771650658641.webp)

---

## How to Run

```bash
cd CreditCardFraudDetection
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```
