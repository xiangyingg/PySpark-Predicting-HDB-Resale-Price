# Predicting HDB Resale Prices with PySpark

This project builds a predictive model for Singapore HDB resale prices using **PySpark** and machine learning techniques. The objective is to estimate resale value from housing attributes and evaluate how strongly each feature affects price.

## Objective

Develop a linear regression model to predict HDB resale prices using:

- Town
- Flat type
- Flat model
- Floor area
- Storey range
- Remaining lease

Transactions valued at $1 million or more are excluded from the modelling dataset to reduce distortion from extreme high-value sales.

## Why this matters

HDB resale prices vary based on a combination of location, unit characteristics, and lease tenure. Understanding these relationships helps quantify pricing drivers and provides a practical estimate of resale value in the market.

## Expected outcome

The project delivers:

- A trained regression model for HDB resale price prediction
- Model performance metrics such as RMSE, MAE, and R²
- Insights into which variables have the strongest influence on price
- A reproducible PySpark workflow for feature engineering, modelling, and evaluation

## Files

- `S10227827H_Sim Xiang Ying_DDP_ASG1_AY2210.ipynb` — full analysis, data preparation, modelling, and evaluation
- `data/sg_flat_prices_mod.csv` — HDB resale transaction dataset (2017–2019)

## Workflow

1. Load the dataset into a Spark DataFrame.
2. Explore and clean the data.
3. Prepare selected features for modelling.
4. Encode categorical variables using `StringIndexer` and `OneHotEncoder`.
5. Assemble and scale features with `VectorAssembler` and `StandardScaler`.
6. Train a `pyspark.ml.regression.LinearRegression` model.
7. Evaluate performance and compare results.

## Requirements

- Python 3
- Apache Spark with PySpark
- Jupyter Notebook or JupyterLab

Install dependencies:

```bash
pip install pyspark jupyter
```

## Run the notebook

```bash
jupyter notebook
```

Open `S10227827H_Sim Xiang Ying_DDP_ASG1_AY2210.ipynb` and run the cells in order. The notebook expects the dataset at `data/sg_flat_prices_mod.csv`.
