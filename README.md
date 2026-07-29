# Predictive Telecom Customer-Churn Analysis using ML

Predicting which telecom customers are about to leave — and figuring out why. This project explores the data, tests six machine-learning models to flag churn risk, groups customers into behavioural segments, and ends with practical, data-backed retention recommendations.

---

## Why this matters

For a telecom operator, churn is the quiet drain on growth: keeping an existing customer is far cheaper than winning a new one, so even a small drop in churn moves the bottom line. The goal here isn't just to build a model that scores well — it's to understand *where* churn concentrates and *what you could actually do about it*.

The dataset is the well-known **IBM Telco Customer Churn** set: **7,043 customers**, 21 attributes covering demographics, the services each customer subscribes to, their account/contract details, and whether they churned. The overall churn rate is **26.5%** — roughly one in four.

---

## What's in the analysis

The notebook works through four stages:

**1. Exploratory data analysis (EDA)**
Cleaning the data (fixing `TotalCharges`, encoding yes/no fields) and profiling how churn varies across contract type, tenure, monthly charges, and combinations of the two.

**2. Statistical feature selection**
Ranking predictors with the right test for each variable type — point-biserial correlation for numeric features, the phi coefficient for binary ones, and Cramér's V for categorical ones — cross-checked against the **Boruta** algorithm.

**3. Supervised classification**
Training and comparing six classifiers on a stratified train/test split to predict churn.

**4. Unsupervised segmentation**
Using K-means (with the elbow method) and hierarchical clustering to group customers into behavioural segments based on tenure and spend.

---

## Key findings

- **Contract type is the single biggest driver.** Month-to-month customers churn at **42.7%**, versus **11.3%** on one-year and just **2.8%** on two-year contracts — and month-to-month is 55% of the base.
- **The first year is the danger zone.** About **53%** of customers churn within their first six months, and nearly half of all churn happens in year one. Tenure is the strongest numeric predictor (point-biserial *r* = −0.35).
- **Payment method and add-ons matter.** Electronic-check users churn at **45%** (vs ~16% on autopay), and customers without online security or tech support churn at ~42%.
- **Risk compounds.** High-charge, month-to-month customers churn at **52%**; low-charge, two-year customers at under **1%**.

---

## Model results

Six models compared on the held-out test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.80** | 0.67 | 0.52 | **0.59** | **0.85** |
| Random Forest | 0.79 | 0.63 | 0.49 | 0.55 | 0.82 |
| Gaussian Naive Bayes | 0.67 | 0.44 | **0.85** | 0.58 | 0.82 |
| SVC | 0.80 | 0.67 | 0.47 | 0.55 | 0.80 |
| K-Nearest Neighbours | 0.77 | 0.58 | 0.54 | 0.56 | 0.78 |
| Decision Tree | 0.73 | 0.48 | 0.51 | 0.49 | 0.65 |

**Logistic regression** was the best all-rounder (0.85 ROC-AUC) and is easy to interpret and deploy. The catch is recall: at the default threshold it catches only ~52% of churners, so for a real retention use-case you'd lower the threshold (or lean on the high-recall Naive Bayes model) — a false alarm just costs a discount, while a miss costs a customer.

---

## Customer segments

Clustering surfaced three clear groups:

| Segment | Profile | Churn rate | Share of base |
|---|---|---|---|
| **New high-spend** | ~15 months, ~$79/mo | **47%** | 34% |
| Budget low-spend | ~27 months, ~$30/mo | 16% | 36% |
| Loyal high-value | ~59 months, ~$91/mo | 15% | 30% |

The **new high-spend** segment is the priority — it's both the largest slice of the base and by far the most likely to leave.

---

## Recommendations (in brief)

The findings point to five retention moves: convert month-to-month customers to term contracts, invest in the first-90-day onboarding experience, stand up a save programme for the new high-spend segment, remove payment and value friction (autopay migration, bundling add-ons), and deploy the churn model as a weekly early-warning score. A full slide deck with these recommendations and supporting sources is included in the repo.

---

## Tech stack

- **Python** — pandas, NumPy
- **scikit-learn** — models, preprocessing, metrics
- **SciPy** — statistical tests
- **Boruta** — feature selection
- **Matplotlib / Seaborn** — visualisation
- **Jupyter Notebook**

---

## Repo structure

```
├── TelcoChurn_cleaned.csv        # cleaned dataset
├── Telco_Churn_ML_Assignment.ipynb   # full analysis notebook
├── Telco_Churn_Insights_Recommendations.pptx   # insights & recommendations deck
└── README.md
```

---

## Getting started

```bash
# clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# install dependencies
pip install pandas numpy scikit-learn scipy boruta matplotlib seaborn jupyter

# launch the notebook
jupyter notebook
```

Then open the notebook and run the cells top to bottom. (Update the CSV path near the top if you move the data file.)

---

## Notes & next steps

This analysis stops at describing and predicting churn — the natural next step is prescriptive work: attaching costs and expected savings to each recommendation, and A/B testing the retention offers rather than assuming they land. Class imbalance (26.5% churn) is also worth revisiting with resampling or class weights to push recall up.

---

*Dataset: IBM Sample Data Sets — Telco Customer Churn.*
