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
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax2)
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
    
    # Weight variance by species (bar chart)
    ax4 = axes[1, 1]
    weight_variance = df.groupby('Species')['Weight'].var().sort_values(ascending=True)
    bars = ax4.barh(weight_variance.index, weight_variance.values, color='steelblue', alpha=0.7)
    ax4.set_xlabel('Weight Variance (g²)')
    ax4.set_ylabel('Species')
    ax4.set_title('Weight Variance by Species')
    # Add value labels on bars
    for bar, val in zip(bars, weight_variance.values):
        ax4.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}', va='center', fontsize=8)
    
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
    
    ### Feature Analysis
    The three length measurements help capture different body proportions:
    - **Length1, Length2, Length3**: Multiple linear dimensions for robust volume estimation
    - **Length3** has highest correlation with Weight (0.923)
    - **Height, Width**: Cross-sectional dimensions
    
    ### Target Variable
    The target variable is **Weight**, representing the fish's weight in grams.
    
    Note: The exact measurement protocol isn't documented in the Kaggle source.
    
    | Issue | Action |
    |-------|--------|
    | Zero weight | Kept - model learns near-zero for small fish |
    | Categorical | Label Encoding |
    | Feature scales | StandardScaler |
    | Non-linear | Random Forest |
    
    ## 4️⃣ Modeling
    
    ### Why Random Forest?
    - **Non-linear relationships**: Fish weight scales with volume (L×W×H), not linearly
    - **Robust to outliers**: Handles measurement errors gracefully
    - **No extensive tuning**: Works well with default parameters
    
    ### Why NOT Linear Regression?
    - Assumes linear relationships: `Weight = a×Length + b`
    - Fish volume relationships are cubic: `Volume ∝ L³`
    - Would underfit - expect much lower R² (~0.7-0.8)
    
    ### Why NOT Multiple Linear Regression?
    - Same issue as above - linear in parameters
    - Doesn't capture interaction effects between features
    - Less accurate for biological growth patterns
    
    ### Why NOT Support Vector Regression (SVR)?
    - Good for small datasets but requires careful parameter tuning
    - Sensitive to feature scaling (we already use StandardScaler)
    - Slower training, less interpretable feature importance
    - Risk of overfitting with inappropriate kernel choice
    
    ## 5️⃣ Evaluation
    - MAE: 44.5g, R²: 0.967 (96.7% variance explained)
    
    ## 6️⃣ Deployment & Real-World Applications
    **Use Cases:**
    - **Fisheries Management**: Estimate stock biomass without sacrificing fish for weighing
    - **Aquaculture Monitoring**: Track growth rates of farmed fish over time
    - **Climate Impact Studies**: Compare fish sizes across years to detect environmental changes
    - **Market Pricing**: Provide fair prices based on accurate weight estimates
    
    **Longitudinal Analysis**: By collecting measurements annually, researchers can:
    - Track growth trends across seasons/generations
    - Identify environmental pressure changes (temperature, food availability)
    - Detect anomalies in fish development patterns
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
        st.session_state['last_prediction'] = prediction
        st.success(f"Predicted Weight: **{prediction:.2f} grams**")
    
    st.divider()
    
    st.subheader("📊 Species Weight Ranges")
    st.markdown("For each species: minimum, average, and maximum recorded weights")
    
    # Get weight ranges for each species
    weight_ranges = df.groupby('Species').agg({
        'Weight': ['min', 'mean', 'max'],
        'Length1': 'mean', 'Length2': 'mean', 'Length3': 'mean',
        'Height': 'mean', 'Width': 'mean'
    }).round(1)
    weight_ranges.columns = ['Min (g)', 'Avg (g)', 'Max (g)', 
                              'Avg L1', 'Avg L2', 'Avg L3', 'Avg H', 'Avg W']
    
    # Display as cards for each species
    cols = st.columns(3)
    for idx, (sp, row) in enumerate(weight_ranges.iterrows()):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"**{sp}**")
                st.markdown(f"📏 Min: **{row['Min (g)']:.0f}g**")
                st.markdown(f"📊 Avg: **{row['Avg (g)']:.0f}g**")
                st.markdown(f"📐 Max: **{row['Max (g)']:.0f}g**")
                st.caption(f"L3: {row['Avg L3']:.1f}cm | H: {row['Avg H']:.1f}cm")
    
    # Show prediction vs range comparison
    if 'last_prediction' in st.session_state and st.session_state['last_prediction']:
        st.divider()
        st.subheader("🎯 Prediction vs Expected Range")
        selected_sp_data = weight_ranges.loc[species]
        pred = st.session_state['last_prediction']
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Your Prediction", f"{pred:.0f}g")
            avg_diff = pred - selected_sp_data['Avg (g)']
            if avg_diff >= 0:
                st.info(f"+{avg_diff:.0f}g from average")
            else:
                st.info(f"{avg_diff:.0f}g from average")
        
        with col2:
            min_w = selected_sp_data['Min (g)']
            max_w = selected_sp_data['Max (g)']
            if min_w <= pred <= max_w:
                position = (pred - min_w) / (max_w - min_w)
                st.progress(position, text=f"Within {species} range: {min_w:.0f}g - {max_w:.0f}g")
            else:
                st.warning(f"Outside {species} range ({min_w:.0f}g - {max_w:.0f}g)")
    
    st.divider()
    
    st.subheader("📈 Detailed Species Statistics")
    species_stats_display = weight_ranges.drop(columns=['Avg L1', 'Avg L2', 'Avg L3', 'Avg H', 'Avg W'])
    st.dataframe(species_stats_display.style.format({'Min (g)': '{:.0f}', 'Avg (g)': '{:.0f}', 'Max (g)': '{:.0f}'}))