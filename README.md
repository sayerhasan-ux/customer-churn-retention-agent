# Customer Churn Prediction & Retention Agent

An end-to-end data science and machine learning project on the Telco Customer Churn dataset: data cleaning, exploratory analysis, statistical testing, customer segmentation, churn modelling, SHAP explanations, a FastAPI prediction service and a LangGraph agent that recommends a retention action.

## Problem

Winning a new customer costs far more than keeping an existing one, but a subscription company rarely knows which customers will leave or why. This project answers three questions: who is likely to leave, why, and what should a retention team do first.

## Dataset

- **Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset on Kaggle). It is a public sample dataset, not live company data.
- **Size:** 7,043 customers, 21 columns (demographics, services, contract, billing, churn label).
- **Baseline churn rate:** 26.5%. Predicting "stays" for everyone gives 73.5% accuracy and catches no churners, so models are judged on precision, recall and ROC-AUC, not accuracy.
- The CSV is not committed. Download it from Kaggle and place it in `data/`.

## What was built

| Stage | What was done |
|---|---|
| Cleaning | Converted `TotalCharges` from text to numeric; 11 blanks belong to new customers with tenure 0 and were filled with 0 |
| Features | `tenure_group`, `avg_monthly_spend`, `num_services` |
| EDA and tests | Churn rate by contract, tenure, payment method and internet service; chi-square tests |
| Segmentation | KMeans (4 clusters) on scaled tenure, monthly charges and number of services |
| Modelling | Logistic Regression (plain and class-weighted) and XGBoost, stratified 80/20 split |
| Explainability | SHAP summary plot and per-customer reasons |
| Serving | FastAPI `/predict` endpoint with Pydantic validation |
| Agent | LangGraph graph that calls the API and chooses a retention action |

## Key findings

| Factor | Churn rate |
|---|---|
| Month-to-month contract | 42.7% (one year 11.3%, two year 2.8%) |
| First 12 months of tenure | about 47% (49+ months about 10%) |
| Fiber optic internet | about 42% (DSL about 19%, no internet about 7%) |
| Electronic check payment | about 45% (other methods about 15-19%) |

Chi-square tests showed significant associations between churn and contract type, internet service and payment method (all p < 0.001).

**Payment method is partly confounded by contract type.** 78% of electronic-check customers are on month-to-month contracts, versus 36-38% of automatic-payment customers. Among month-to-month customers only, electronic check still churns more (53.7% vs 31.6-34.1%), but the gap shrinks from about 3x to about 1.6x. These are associations, not proof of cause.

### Customer segments

| Segment | Customers | Avg tenure (months) | Churn rate | Churned |
|---|---|---|---|---|
| At-Risk Mid-Tier | 2,315 | 15 | 47% | 1,099 |
| New & Basic | 1,557 | 10 | 24% | 377 |
| Loyal Premium | 2,144 | 57 | 16% | 344 |
| Loyal Budget | 1,027 | 55 | 5% | 49 |

The At-Risk Mid-Tier segment holds 33% of customers but about 59% of all churners (1,099 of 1,869).

## Model results

Evaluated on 1,409 held-out customers (374 churners). Precision, recall and F1 are for the churn class.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (plain) | 0.655 | 0.519 | 0.579 | 0.843 |
| Logistic Regression (balanced) | 0.500 | 0.794 | 0.614 | 0.842 |
| XGBoost (balanced) | 0.511 | 0.813 | 0.627 | 0.842 |

- All three models reach a ROC-AUC of about 0.84, so they rank churn risk equally well.
- Class weighting raised recall from 0.52 to about 0.79-0.81 at the cost of precision (0.66 down to about 0.50). It changes the precision/recall trade-off; it does not make the model better at separating customers.
- XGBoost gives only a marginal gain over balanced Logistic Regression, suggesting the signal in this dataset is mostly simple.
- Which model to prefer is a business decision: if missing a churner costs far more than a wasted retention offer, favour the higher-recall models.

SHAP agrees with the exploratory analysis: short tenure, month-to-month contracts, fiber optic internet and electronic check payment are the strongest churn drivers.

## API

`POST /predict` takes a customer record and returns the churn probability, a risk label and the top reasons from SHAP. Inputs are validated with Pydantic (`Literal` types for categories, `0 <= tenure <= 72`), so invalid values return a `422` error.

Example response for a new month-to-month fiber customer paying by electronic check:

```json
{
  "churn_probability": 0.833,
  "risk": "high",
  "reasons_to_leave": [
    {"feature": "tenure", "impact": 0.595},
    {"feature": "Contract_Two year", "impact": 0.276},
    {"feature": "InternetService_Fiber optic", "impact": 0.275}
  ],
  "reasons_to_stay": [
    {"feature": "MultipleLines_Yes", "impact": -0.118},
    {"feature": "avg_monthly_spend", "impact": -0.082}
  ]
}
```

Reason names come from one-hot columns. `Contract_Two year` with a positive impact means the customer does **not** have a two-year contract. `impact` is in log-odds units.

## Retention agent

`src/agent.py` is a LangGraph graph:

```
START -> predict -> (high risk) -> retain    -> END
                 -> (low risk)  -> no_action -> END
```

- `predict` calls the API; the model alone produces the probability.
- A conditional edge routes high-risk customers to `retain` and low-risk customers to `no_action`.
- `retain` picks the first risk factor that has an action in a hand-written rules table and builds a message.

Example output:

```
Churn risk is high (83%). Main risk factors: short tenure, no 2-year contract, fiber optic service. Recommended action: Schedule an onboarding check-in call (customer is still new).
```

The agent does not use an LLM yet, so it needs no API key. The action table is a set of business rules written by hand, not something the model learned.

## How to run

```bash
git clone https://github.com/sayerhasan-ux/customer-churn-retention-agent.git
cd customer-churn-retention-agent
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost shap joblib jupyter fastapi uvicorn langgraph requests
```

1. Download the dataset from Kaggle into `data/`.
2. Open `notebooks/01_cleaning_eda.ipynb` and use **Restart, then Run All**.
3. Start the API (the trained model files are included in `src/`):
   ```bash
   cd src
   uvicorn api:app --reload
   ```
   Interactive docs: http://127.0.0.1:8000/docs
4. In a second terminal, run the agent:
   ```bash
   cd src
   python agent.py
   ```

## Repository structure

```
data/         raw CSV (not committed)
notebooks/    01_cleaning_eda.ipynb  (cleaning, EDA, tests, segmentation, models, SHAP)
src/          api.py, agent.py, churn_model.joblib, model_columns.joblib
```

## Limitations

- Public sample dataset with no timestamps, so there is no time-based validation and no real-world deployment testing.
- Findings are associations. They do not prove that changing a contract or payment method would change churn.
- `tenure`, `TotalCharges` and `avg_monthly_spend` are strongly related, so their individual SHAP contributions overlap.
- The `high` risk threshold is a fixed 0.5 and was not tuned against the cost of a missed churner versus a retention offer.
- Hyperparameters were set by hand and not cross-validated.
- The API validates each field separately but not relationships between fields (for example, `tenure` 72 with `TotalCharges` 0 is accepted).
- Retention actions are rule-based, not learned.

## Planned next steps

- Tune the decision threshold with explicit costs.
- Add cross-field validation to the API.
- Use an LLM to write the retention message, with the model still producing the probability.
- Add automated tests for the API and agent.