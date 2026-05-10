# Laptop Price Prediction

A supervised machine learning project that predicts laptop prices based on hardware specifications. Built an end-to-end pipeline covering data preprocessing, feature selection, model training, and hyperparameter tuning. Achieved an R² of **0.84** on the test set.

## Tech Stack

Python · scikit-learn · pandas · NumPy · seaborn · matplotlib

## What I Did

- Cleaned and preprocessed a 1,275-row dataset (min-max scaling, Box-Cox normalization, one-hot encoding), expanding raw features into 875 encoded columns
- Applied **mutual information regression** to reduce dimensionality from 875 to 12 most predictive features (RAM, CPU frequency, storage type, screen resolution, etc.)
- Trained and compared **Linear Regression, Random Forest, and Gradient Boosting** models using 5-fold cross-validation
- Tuned hyperparameters via **GridSearchCV** and explored **stacking ensembles**, identifying Gradient Boosting as the top performer

## Results

| Model | R² |
|---|---|
| Linear Regression | 0.71 |
| Random Forest | 0.82 |
| **Gradient Boosting (tuned)** | **0.84** |