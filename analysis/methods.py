"""
Fishy Predictions - Methods Documentation

This file contains alternative model implementations and documentation
for educational purposes.

Alternative models for comparison:
1. LinearRegression - Simple, interpretable baseline
2. SVR - Good for small datasets with careful scaling
"""

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def get_feature_importance(model, feature_names):
    """Get feature importance from trained model."""
    return dict(zip(feature_names, model.feature_importances_))


def train_linear_regression(X_train, y_train):
    """
    Linear Regression baseline - simple, interpretable but lower accuracy.
    
    Use case: When you need a simple baseline or interpretable coefficients.
    Limitation: Assumes linear relationships between features and target.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_svr(X_train, y_train):
    """
    Support Vector Regression - good for small datasets.
    
    Use case: Small datasets with clear decision boundaries.
    Limitation: Sensitive to feature scaling, slower training.
    """
    model = SVR(kernel='rbf', C=100, gamma=0.1)
    model.fit(X_train, y_train)
    return model


def train_xgboost_comparison(X_train, y_train):
    """
    Placeholder for XGBoost - often better accuracy than Random Forest.
    
    Note: XGBoost not included to keep dependencies minimal.
    Would provide higher accuracy but requires additional installation.
    """
    pass