import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def prepare_features(df: pd.DataFrame):
    le = LabelEncoder()
    df['Species_encoded'] = le.fit_transform(df['Species'])
    
    features = ['Species_encoded', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
    X = df[features]
    y = df['Weight']
    
    return X, y, le

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return model, scaler, metrics

def predict_weight(model, scaler, le, species: str, length1: float, length2: float, 
                   length3: float, height: float, width: float):
    species_encoded = le.transform([species])[0]
    feature_names = ['Species_encoded', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
    features = pd.DataFrame(
        [[species_encoded, length1, length2, length3, height, width]],
        columns=feature_names
    )
    features_scaled = scaler.transform(features)
    return model.predict(features_scaled)[0]