# Predicting HDB Resale Prices with PySpark

This school assignment uses **PySpark machine learning** to study and predict Singapore HDB resale flat prices. The project uses Spark DataFrames and `pyspark.ml`; it does not use pandas for data processing.

## Project objective

Build a linear regression model to estimate HDB resale prices from:

- Town
- Flat type
- Flat model
- Floor area
- Storey range
- Remaining lease

High-priced transactions of $1 million or more are excluded from the modelling analysis.

## Project files

- `S10227827H_Sim Xiang Ying_DDP_ASG1_AY2210.ipynb` — complete analysis, transformation, modelling, and evaluation
- `data/sg_flat_prices_mod.csv` — HDB resale transaction dataset from 2017 to 2019

## PySpark workflow

1. Create a `SparkSession` and load the CSV into a Spark DataFrame.
2. Explore the data using Spark SQL/DataFrame operations.
3. Clean missing values and prepare the selected features.
4. Encode categorical columns with `StringIndexer` and `OneHotEncoder`.
5. Assemble and scale features with `VectorAssembler` and `StandardScaler`.
6. Train a `pyspark.ml.regression.LinearRegression` model.
7. Evaluate the predictions using regression metrics and select the final model.

## Requirements

- Python 3
- Apache Spark with PySpark
- Jupyter Notebook or JupyterLab

Install the main Python dependencies with:

```bash
pip install pyspark jupyter
```

## Running the notebook

From the project directory:

```bash
jupyter notebook
```

Open `S10227827H_Sim Xiang Ying_DDP_ASG1_AY2210.ipynb` and run the cells in order. The notebook expects the dataset at `data/sg_flat_prices_mod.csv`.
