# Comparison Between Classical Curve Fitting Methods & MLPs
Implementing different traditional Curve Fitting methods and MLPs-Multilayer Perceptrons and Making Comparison between them

## Overview
In this project, we will make comparisons between some classical curve fitting methods, specifically polynomial interpolation, (cubic) spline interpolation, and polynomial regression, and the use of MLPs (Multi-Layer Perceptrons), a type of ANNs.
Our main objective is to assess whether MLPs can provide a better generalization and accuracy than the classical approaches.

## Features
- Synthetic Data Generation: Generate a dataset based on a function, add noise to the datapoints, scale them to the range [0, 1], and save them to 3 CSV files.
- Model Building: Building the algorithm (lagrange, cubic spline, regression, MLP) to predict
- Visualization: Visualize predicted data and true test data

## Files
- `datageneration.py`: generates synthetic data from the function f(x) = e^(-0.25x)cos(2x)
- `lagrange.py`: implements the Lagrange interpolation method. Note that due to the nature of Lagrange interpolation, only a small subset of datapoints are chosen to build the model.
- `cubicspline.py`: implements the Cubic Spline interpolation method.
- `polynomialregression.py`: implements the Polynomial Regression method using `PolynomialFeatures`, features polynomials of degree 1~10
- `mlp.py`: implement the MLP model, features 4 variants of MLP: Basic (default) MLP, MLP with Batch Norm + Leaky ReLU, MLP with Early Stopping, and MLP with LR Scheduler.
- `mlp_residual.py`: implements the Residual MLP model.

## How to run
1. Open terminal and install required packages: `pip install -r requirements.txt`. Make sure that you have installed Python 3.13 on your computer.
2. Run the Python file from terminal using the following code:
- `python datageneration.py` (Optional as there are already the dataset folders)
- `python main.py`

## Group 23
- Nguyễn Huy Diễn - 20235910
- Trần Trung Hiếu - 20235934
- Phạm Song Hào - 20235930
- Nguyễn Tiến Đạt - 20239715
- Lê Hoàng Kiên - 20235958
- Phạm Đức Phước - 20235988