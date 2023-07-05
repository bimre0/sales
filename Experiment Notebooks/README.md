# Market Sales Data Analysis

This repository contains code and resources for analyzing market sales data. The goal of this project is to provide insights and visualizations to understand sales trends, customer behavior, and product performance.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In today's business landscape, analyzing market sales data is crucial for making informed decisions and developing effective strategies. This project aims to facilitate the analysis process by providing a set of tools and scripts to explore, clean, visualize, and extract insights from market sales data. The project has 2 categories; Experiments(notebook files) and Scripts(python files)

## Assumptions

1. The input data for training the model is stored in  `train.parquet`.
2. The file contains the following columns: ['date', 'store_nbr', 'family', 'sales', 'onpromotion']
3. The target variable is a `sales`
4. The data does not contain any missing values.
5. The data does not contain any outliers.
6. The data has been preprocessed and scaled appropriately.
7. The model will be trained using a logistic regression algorithm. The model can be swapped with more complex models. But in order to keep it simple and prototype fast, it was chosen
8. The trained model will be saved as a pickle file named `model.pkl`.
9. The model's performance will be evaluated using RMSE, MAE, R2 as the metric.
10. The evaluation results will be saved in a text file named `evaluation_results.txt`.
11. The project dependencies are listed in the `requirements.txt` file.
12. The project can be executed by running the `train.py` script.
13. The hyperparameters can be set inside `params.yaml` file

## Usage

1. Install the project dependencies by running `pip install -r requirements.txt`.
2. Place the input data file `train.py` in the project directory.
3. Run the `train.py` script to train the model and evaluate its performance.
4. The trained model will be saved as `model.pkl` and the evaluation results will be saved in `evaluation_results.txt`.

