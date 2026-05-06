import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from analysis.model import load_data, prepare_features, train_model, predict_weight

st.set_page_config(page_title="Fishy Predictions", page_icon="🐟", layout="wide")

@st.cache_resource
def init_model():
    data_path = Path(__file__).parent / "data_source" / "fish_weight.csv"
    df = load_data(data_path)
    X, y, le = prepare_features(df)
    model, scaler, metrics = train_model(X, y)
    return df, model, scaler, le, metrics

df, model, scaler, le, metrics = init_model()
st.session_state['df'] = df
st.session_state['model'] = model
st.session_state['scaler'] = scaler
st.session_state['le'] = le

# Top navigation tabs
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📊 CRISP-DM", "🎯 Predict"])

with tab1:
    st.title("🐟 Fishy Predictions")
    st.caption("by Marcell_JW (github: mjwsolver)")
    
    st.markdown(f"""
    Fishy Predictions is a machine learning powered application that predicts the weight of fish 
    based on their physical characteristics.
    
    ### Model Performance
    - MAE: {metrics['mae']:.2f} grams
    - R² Score: {metrics['r2']:.3f}
    """)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

with tab2:
    st.title("🔬 CRISP-DM: Data Science Methodology")
    
    st.markdown("""
    ## Cross-Industry Standard Process for Data Mining (CRISP-DM)
    
    ## 1️⃣ Business Understanding
    **Objective:** Predict fish weight from physical measurements.
    **Success Criteria:** R² > 0.9, understandable model, practical deployment.
    """)
    
    st.markdown("""
    ## 2️⃣ Data Understanding
    """)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Weight distribution by species
    ax1 = axes[0, 0]
    df.boxplot(column='Weight', by='Species', ax=ax1)
    ax1.set_title('Weight Distribution by Species')
    ax1.set_xlabel('Species')
    ax1.set_ylabel('Weight (g)')
    plt.suptitle('')
    
    # Correlation heatmap
    ax2 = axes[0, 1]
    numeric_cols = ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
    ax2.set_title('Feature Correlation Matrix')
    
    # Length vs Weight scatter
    ax3 = axes[1, 0]
    species_list = df['Species'].unique()
    colors = plt.cm.tab10(range(len(species_list)))
    for i, species in enumerate(species_list):
        species_data = df[df['Species'] == species]
        ax3.scatter(species_data['Length3'], species_data['Weight'], 
                    label=species, color=colors[i], alpha=0.7)
    ax3.set_xlabel('Length3 (cm)')
    ax3.set_ylabel('Weight (g)')
    ax3.set_title('Length3 vs Weight by Species')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Species Count", df['Species'].nunique())
    with col2:
        st.metric("Missing Values", df.isnull().sum().sum())
        st.metric("Zero Weights", (df['Weight'] == 0).sum())
    
    st.markdown("""
    ## 3️⃣ Data Preparation
    | Issue | Action |
    |-------|--------|
    | Zero weight | Kept - model learns near-zero for small fish |
    | Categorical | Label Encoding |
    | Feature scales | StandardScaler |
    | Non-linear | Random Forest |
    
    ## 4️⃣ Modeling
    **Why Random Forest?**
    - Non-linear relationships (volume-based weight)
    - Robust to outliers
    - No extensive tuning needed
    
    ## 5️⃣ Evaluation
    - MAE: 44.5g, R²: 0.967 (96.7% variance explained)
    """)

with tab3:
    st.title("🎯 Predict Fish Weight")
    
    species_list = ['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt']
    
    col1, col2 = st.columns(2)
    with col1:
        species = st.selectbox("Species", species_list)
        length1 = st.number_input("Length1 (vertical)", min_value=0.0, value=25.0, step=0.1)
        length2 = st.number_input("Length2 (diagonal)", min_value=0.0, value=27.0, step=0.1)
        length3 = st.number_input("Length3 (cross)", min_value=0.0, value=29.0, step=0.1)
    with col2:
        height = st.number_input("Height", min_value=0.0, value=7.0, step=0.1)
        width = st.number_input("Width", min_value=0.0, value=3.0, step=0.1)
    
    if st.button("🔮 Predict Weight"):
        prediction = predict_weight(model, scaler, le, species, length1, length2, length3, height, width)
        st.success(f"Predicted Weight: **{prediction:.2f} grams**")
    
    st.divider()
    st.subheader("📊 Sample Predictions by Species")
    st.markdown("Average measurements and weights for each species type:")
    
    species_stats = df.groupby('Species').agg({
        'Weight': ['mean', 'min', 'max', 'count'],
        'Length1': 'mean', 'Length2': 'mean', 'Length3': 'mean',
        'Height': 'mean', 'Width': 'mean'
    }).round(2)
    species_stats.columns = ['Avg Weight (g)', 'Min', 'Max', 'Count', 
                              'Avg L1', 'Avg L2', 'Avg L3', 'Avg Height', 'Avg Width']
    st.dataframe(species_stats.style.format({'Avg Weight (g)': '{:.1f}'}))