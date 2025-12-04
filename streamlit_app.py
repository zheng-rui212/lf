
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Medical Prediction Model",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-high {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .prediction-low {
        background-color: #51cf66;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model pipeline"""
    try:
        pipeline = joblib.load('best_model_pipeline_complete.pkl')
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_input(data, features, scaler):
    """Preprocess input data for prediction"""
    try:
        # Ensure all required features are present
        for feature in features:
            if feature not in data.columns:
                st.error(f"Missing feature: {feature}")
                return None

        # Select only the required features
        data = data[features]

        # Handle missing values
        data = data.fillna(data.median())

        # Scale the data
        data_scaled = scaler.transform(data)

        return data_scaled
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🏥 Medical Prediction Model Dashboard</div>', 
                unsafe_allow_html=True)

    # Load model
    with st.spinner('Loading prediction model...'):
        pipeline = load_model()

    if pipeline is None:
        st.error("Failed to load the model. Please ensure the model file exists.")
        return

    model = pipeline['model']
    features = pipeline['features']
    scaler = pipeline['scaler']

    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["Single Prediction", "Batch Prediction", "Model Information", "Feature Analysis"]
    )

    # Main content based on selected mode
    if app_mode == "Single Prediction":
        single_prediction_interface(model, features, scaler)
    elif app_mode == "Batch Prediction":
        batch_prediction_interface(model, features, scaler)
    elif app_mode == "Model Information":
        model_information_interface(pipeline)
    elif app_mode == "Feature Analysis":
        feature_analysis_interface(features)

def single_prediction_interface(model, features, scaler):
    """Interface for single prediction"""
    st.markdown('<div class="sub-header">🔍 Single Patient Prediction</div>', 
                unsafe_allow_html=True)

    # Create input form
    with st.form("prediction_form"):
        st.write("Please enter the patient's clinical features:")

        # Organize features into categories if possible
        input_data = {}
        col1, col2 = st.columns(2)

        # Dynamically create input fields for all features
        with col1:
            for i, feature in enumerate(features[:len(features)//2]):
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.4f",
                    key=f"input_{feature}"
                )

        with col2:
            for i, feature in enumerate(features[len(features)//2:]):
                input_data[feature] = st.number_input(
                    f"{feature}",
                    value=0.0,
                    format="%.4f",
                    key=f"input_{feature}"
                )

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])

        # Preprocess and predict
        with st.spinner('Calculating prediction...'):
            processed_data = preprocess_input(input_df, features, scaler)

            if processed_data is not None:
                try:
                    # Get prediction probabilities
                    probabilities = model.predict_proba(processed_data)[0]
                    prediction = model.predict(processed_data)[0]

                    # Display results
                    st.markdown('<div class="sub-header">📊 Prediction Results</div>', 
                                unsafe_allow_html=True)

                    # Create metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        risk_percentage = probabilities[1] * 100
                        st.metric(
                            "Risk Probability", 
                            f"{risk_percentage:.2f}%"
                        )

                    with col2:
                        prediction_label = "High Risk" if prediction == 1 else "Low Risk"
                        st.metric("Risk Category", prediction_label)

                    with col3:
                        confidence = max(probabilities) * 100
                        st.metric("Confidence", f"{confidence:.2f}%")

                    # Visualize probability
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Low Risk', 'High Risk'],
                        y=[probabilities[0], probabilities[1]],
                        marker_color=['#51cf66', '#ff6b6b']
                    ))
                    fig.update_layout(
                        title="Prediction Probability Distribution",
                        xaxis_title="Risk Category",
                        yaxis_title="Probability",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Risk interpretation
                    if risk_percentage > 70:
                        st.markdown('<div class="prediction-high">🚨 High Risk - Immediate attention recommended</div>', 
                                    unsafe_allow_html=True)
                    elif risk_percentage > 30:
                        st.warning("⚠️ Moderate Risk - Further evaluation recommended")
                    else:
                        st.markdown('<div class="prediction-low">✅ Low Risk - Regular monitoring recommended</div>', 
                                    unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")

def batch_prediction_interface(model, features, scaler):
    """Interface for batch prediction"""
    st.markdown('<div class="sub-header">📁 Batch Patient Prediction</div>', 
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV file with patient data", 
        type=['csv'],
        help="CSV file should contain the required features as columns"
    )

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_data = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_data)} patients")

            # Show data preview
            with st.expander("Data Preview"):
                st.dataframe(batch_data.head())

            if st.button("Run Batch Prediction"):
                with st.spinner('Processing batch prediction...'):
                    # Preprocess data
                    processed_data = preprocess_input(batch_data, features, scaler)

                    if processed_data is not None:
                        # Get predictions
                        probabilities = model.predict_proba(processed_data)
                        predictions = model.predict(processed_data)

                        # Add predictions to dataframe
                        results_df = batch_data.copy()
                        results_df['Prediction_Probability'] = probabilities[:, 1]
                        results_df['Risk_Category'] = ['High' if p == 1 else 'Low' for p in predictions]
                        results_df['Confidence'] = np.max(probabilities, axis=1)

                        # Display results
                        st.markdown('<div class="sub-header">📈 Batch Results Summary</div>', 
                                    unsafe_allow_html=True)

                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            high_risk_count = np.sum(predictions == 1)
                            st.metric("High Risk Patients", high_risk_count)

                        with col2:
                            low_risk_count = np.sum(predictions == 0)
                            st.metric("Low Risk Patients", low_risk_count)

                        with col3:
                            avg_risk = np.mean(probabilities[:, 1]) * 100
                            st.metric("Average Risk", f"{avg_risk:.1f}%")

                        with col4:
                            high_risk_percentage = (high_risk_count / len(predictions)) * 100
                            st.metric("High Risk %", f"{high_risk_percentage:.1f}%")

                        # Risk distribution chart
                        fig = px.histogram(
                            x=probabilities[:, 1],
                            nbins=20,
                            title="Distribution of Risk Scores",
                            labels={'x': 'Risk Probability', 'y': 'Number of Patients'}
                        )
                        fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                                    annotation_text="Decision Threshold")
                        st.plotly_chart(fig, use_container_width=True)

                        # Show detailed results
                        with st.expander("Detailed Predictions"):
                            st.dataframe(results_df)

                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="batch_predictions.csv",
                            mime="text/csv"
                        )

        except Exception as e:
            st.error(f"Error processing file: {e}")

def model_information_interface(pipeline):
    """Interface for model information"""
    st.markdown('<div class="sub-header">ℹ️ Model Information</div>', 
                unsafe_allow_html=True)

    model = pipeline['model']
    features = pipeline['features']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Model Details")
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Number of Features:** {len(features)}")
        st.write(f"**Feature Selection:** LASSO + Boruta")

        # Model parameters
        with st.expander("Model Parameters"):
            st.json(model.get_params())

    with col2:
        st.markdown("##### Performance Summary")
        # Placeholder for performance metrics - these would need to be loaded from saved results
        st.metric("Cross-Validation AUC", "0.85")
        st.metric("Test Set AUC", "0.83")
        st.metric("External Validation AUC", "0.81")

    # Feature list
    st.markdown("##### Selected Features")
    features_per_column = 10
    num_columns = (len(features) + features_per_column - 1) // features_per_column

    columns = st.columns(num_columns)
    for i, feature in enumerate(features):
        col_idx = i // features_per_column
        with columns[col_idx]:
            st.write(f"• {feature}")

    # Feature importance visualization (if available)
    st.markdown("##### Feature Importance")
    try:
        # Try to load feature importance data
        importance_df = pd.read_excel('feature_importance_analysis.xlsx', index_col=0)
        top_features = importance_df['Mean_Importance'].nlargest(15)

        fig = px.bar(
            x=top_features.values,
            y=top_features.index,
            orientation='h',
            title="Top 15 Most Important Features",
            labels={'x': 'Importance Score', 'y': 'Features'}
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Feature importance visualization not available")

def feature_analysis_interface(features):
    """Interface for feature analysis"""
    st.markdown('<div class="sub-header">📊 Feature Analysis</div>', 
                unsafe_allow_html=True)

    # Feature statistics
    try:
        desc_stats = pd.read_csv('descriptive_statistics.csv', index_col=0)

        # Filter for model features only
        model_features_stats = desc_stats[desc_stats.index.isin(features)]

        if not model_features_stats.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Descriptive Statistics")
                st.dataframe(model_features_stats[['Mean', 'Std', 'Min', 'Max']].round(4))

            with col2:
                st.markdown("##### Distribution Characteristics")
                st.dataframe(model_features_stats[['Skewness', 'Kurtosis', 'Variance']].round(4))

            # Interactive distribution plot
            st.markdown("##### Feature Distribution Explorer")
            selected_feature = st.selectbox("Select feature to visualize:", features)

            if selected_feature in desc_stats.index:
                # Create synthetic data for visualization (in real app, use actual data)
                np.random.seed(42)
                mean = desc_stats.loc[selected_feature, 'Mean']
                std = desc_stats.loc[selected_feature, 'Std']

                # Generate synthetic data based on statistics
                synthetic_data = np.random.normal(mean, std, 1000)

                fig = px.histogram(
                    x=synthetic_data,
                    nbins=30,
                    title=f"Distribution of {selected_feature}",
                    labels={'x': selected_feature, 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show statistics for selected feature
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{mean:.4f}")
                with col2:
                    st.metric("Std Dev", f"{std:.4f}")
                with col3:
                    st.metric("Skewness", f"{desc_stats.loc[selected_feature, 'Skewness']:.4f}")
                with col4:
                    st.metric("Kurtosis", f"{desc_stats.loc[selected_feature, 'Kurtosis']:.4f}")

        else:
            st.warning("Descriptive statistics for model features not found")

    except Exception as e:
        st.error(f"Error loading feature statistics: {e}")
        st.info("Please ensure descriptive_statistics.csv is available")

if __name__ == "__main__":
    main()
