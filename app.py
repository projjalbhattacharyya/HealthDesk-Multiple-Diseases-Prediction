import os
import numpy as np
import pickle
import joblib
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import altair as alt

# Set page configuration
st.set_page_config(page_title="HealthDesk - Your Medical Assistant",
                   layout="wide",
                   page_icon="⚕️")
    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load Diabetes Models and Preprocessors
diabetes_GBA_model = joblib.load("models/diabetes_GBA_model.pkl")
diabetes_minmax_scaler = joblib.load("models/diabetes_minmax_scaler.pkl")
diabetes_onehot_encoder = joblib.load("models/diabetes_onehot_smoking_and_gender_encoder.pkl")

# Load Heart Disease Models and Preprocessors
heart_disease_scaler = joblib.load("models/heart_disease_scaler.joblib")
heart_disease_svm_model = joblib.load("models/heart_disease_svm_model.joblib")

# Load Lung Cancer Models and Preprocessors
lung_cancer_KNN_model = joblib.load("models/lung_cancer_KNN_model.pkl")
lung_cancer_onehot_encoder = joblib.load("models/lung_cancer_onehot_encoder.pkl")
lung_cancer_standard_scaler = joblib.load("models/lung_cancer_standard_scaler.pkl")

# Load Parkinson's Disease Models and Preprocessors
parkinsons_knn_model = joblib.load("models/Parkinsons_knn_model.joblib")
parkinsons_pca = joblib.load("models/Parkinsons_pca.joblib")
parkinsons_scaler = joblib.load("models/Parkinsons_scaler.joblib")

# Sidebar for Navigation
with st.sidebar:
    selected = option_menu(
        'HealthDesk',
        [
            'Diabetes Prediction',
            'Heart Disease Prediction',
            'Parkinsons Prediction',
            'Lung Cancer Risk Prediction'  # Added Lung Cancer Prediction
        ],
        menu_icon='hospital',
        icons=['activity', 'heart', 'person', 'lungs'],  # Added lung-related icon
        default_index=0
    )

# Define columns used in training
diabetes_categorical_columns = ['gender', 'smoking_history']
diabetes_columns_to_scale = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
diabetes_all_features = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'gender_Male', 'smoking_history_current', 'smoking_history_ever',
    'smoking_history_former', 'smoking_history_never', 'smoking_history_not current'
]

def preprocess_diabetes_data(new_data):
    # One-Hot Encode categorical columns
    encoded_data = diabetes_onehot_encoder.transform(new_data[diabetes_categorical_columns])
    encoded_columns = diabetes_onehot_encoder.get_feature_names_out(diabetes_categorical_columns)
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoded_columns, index=new_data.index)
    
    # Drop original categorical columns and merge encoded data
    new_data = new_data.drop(columns=diabetes_categorical_columns)
    new_data = pd.concat([new_data, encoded_df], axis=1)
    
    # Scale numerical columns
    new_data[diabetes_columns_to_scale] = diabetes_minmax_scaler.transform(new_data[diabetes_columns_to_scale])
    
    # Ensure all features are in the correct order
    missing_cols = set(diabetes_all_features) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0  # Add missing columns with default value
    
    new_data = new_data[diabetes_all_features]  # Ensure correct column order
    return new_data

if selected == "Diabetes Prediction":
    st.title("🏥Multiple Disease Prediction System")
    st.header("🩺Diabetes Prediction")
    
    age = st.number_input("Age", min_value=1, max_value=100, value=1)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0)
    HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=10.0, value=3.0)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=350, value=80)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    smoking_history = st.selectbox("Smoking History", ['never', 'current', 'former', 'ever', 'not current'])
    
    if st.button("Predict Diabetes"):

        hypertension_value = 1 if hypertension == "Yes" else 0
        heart_disease_value = 1 if heart_disease == "Yes" else 0

        new_input_data = pd.DataFrame({
            'age': [age],
            'hypertension': [hypertension_value],
            'heart_disease': [heart_disease_value],
            'bmi': [bmi],
            'HbA1c_level': [HbA1c_level],
            'blood_glucose_level': [blood_glucose_level],
            'gender': [gender],
            'smoking_history': [smoking_history]
        })
        
        processed_data = preprocess_diabetes_data(new_input_data)
        prediction = diabetes_GBA_model.predict(processed_data)
        confidence = diabetes_GBA_model.predict_proba(processed_data)[0]
        
        if prediction[0] == 1:
            st.error("The model predicts that the patient has diabetes.")
        else:
            st.success("The model predicts that the patient does not have diabetes.")

            # Create a dataframe for visualization
        confidence_df = pd.DataFrame({
            "Prediction": ["No Diabetes", "Diabetes"],
            "Confidence": [confidence[0] * 100, confidence[1] * 100]
        })

        # Create Altair bar chart
        chart = alt.Chart(confidence_df).mark_bar().encode(
            x=alt.X("Prediction", title="Prediction Outcome"),
            y=alt.Y("Confidence", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
            color="Prediction"
        ).properties(title="Prediction Confidence Level", width=500)

        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)


# Heart Disease Prediction Page
if selected == "Heart Disease Prediction":
    st.title("🏥Multiple Disease Prediction System")
    st.header("🩺Heart Disease Prediction")

    # Creating two columns for better UI
    col1, col2 = st.columns(2)

    # First Column Inputs
    with col1:
        age = st.number_input("Age", min_value=30, max_value=80, value=30)
        sex = st.radio("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=95, max_value=200, value=95)
        chol = st.number_input("Cholesterol (chol)", min_value=100, max_value=600, value=130)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (FBS)", ["Yes", "No"])

    # Second Column Inputs
    with col2:
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate (thalach)", min_value=70, max_value=250, value=70)
        exang = st.radio("Exercise-Induced Angina (exang)", ["Yes", "No"])
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=0.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise (slope)", [0, 1, 2])
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    # Convert categorical values to numeric
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    # Prediction button
    if st.button("Predict Heart Disease"):
        # Create DataFrame for user input
        input_df = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        ]], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])

        # Scale input data
        input_scaled = heart_disease_scaler.transform(input_df)

        # Make prediction
        prediction = heart_disease_svm_model.predict(input_scaled)

        # Get confidence score using decision_function()
        confidence_score = heart_disease_svm_model.decision_function(input_scaled)
        
        # Convert confidence score to probability-like value
        confidence = (1 / (1 + np.exp(-confidence_score)))[0]

        # Display result
        if prediction[0] == 1:
            st.error("The model predicts that the patient **has heart disease**.")
        else:
            st.success("The model predicts that the patient **does not have heart disease**.")

        # Create Altair bar chart - dataframe for visualization
        confidence_df = pd.DataFrame({
            "Prediction": ["No Heart Disease", "Heart Disease"],
            "Confidence": [100 - confidence * 100, confidence * 100]
        })

        # Create Altair bar chart
        chart = alt.Chart(confidence_df).mark_bar().encode(
            x=alt.X("Prediction", title="Prediction Outcome"),
            y=alt.Y("Confidence", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
            color="Prediction"
        ).properties(title="Prediction Confidence Level", width=500)

        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)


# Parkinson's Prediction Page
# Define the column names based on training features
columns = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
           'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
           'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
           'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
           'spread2', 'D2', 'PPE']

if selected == "Parkinsons Prediction":
    st.title("🏥Multiple Disease Prediction System")
    st.header("🩺Parkinson's Disease Prediction")

    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)

    # Input fields distributed across three columns
    with col1:
        MDVP_Fo_Hz = st.number_input("MDVP:Fo(Hz)", min_value=88.333, max_value=260.105, value=88.333, format="%.3f")
        MDVP_Fhi_Hz = st.number_input("MDVP:Fhi(Hz)", min_value=102.145, max_value=600.00, value=102.145, format="%.3f")
        MDVP_Flo_Hz = st.number_input("MDVP:Flo(Hz)", min_value=65.476, max_value=250.00, value=66.000, format="%.3f")
        MDVP_Jitter = st.number_input("MDVP:Jitter(%)", min_value=0.00168, max_value=0.1, value=0.00168, format="%.5f")
        MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.000007, max_value=0.0003, value=0.000007, format="%.6f")
        MDVP_RAP = st.number_input("MDVP:RAP", min_value=0.00068, max_value=0.03, value=0.00068, format="%.5f")
        MDVP_PPQ = st.number_input("MDVP:PPQ", min_value=0.00092, max_value=0.02, value=0.00092, format="%.5f")
        Jitter_DDP = st.number_input("Jitter:DDP", min_value=0.00204, max_value=0.06433, value=0.06433, format="%.5f")

    with col2:
        MDVP_Shimmer = st.number_input("MDVP:Shimmer", min_value=0.00954, max_value=0.2, value=0.00954, format="%.5f")
        MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.085, max_value=1.5, value=0.085, format="%.3f")
        Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.00455, max_value=0.05647, value=0.00455, format="%.5f")
        Shimmer_APQ5 = st.number_input("Shimmer:APQ5", min_value=0.0057, max_value=0.1, value=0.0057, format="%.4f")
        MDVP_APQ = st.number_input("MDVP:APQ", min_value=0.00719, max_value=0.2, value=0.00719, format="%.5f")
        Shimmer_DDA = st.number_input("Shimmer:DDA", min_value=0.01364, max_value=0.2, value=0.01364, format="%.5f")
        NHR = st.number_input("NHR", min_value=0.00065, max_value=0.4, value=0.00065, format="%.5f")
        HNR = st.number_input("HNR", min_value=8.441, max_value=35.00, value=8.441, format="%.3f")

    with col3:
        RPDE = st.number_input("RPDE", min_value=0.25657, max_value=0.99999, value=0.25657, format="%.5f")
        DFA = st.number_input("DFA", min_value=0.574282, max_value=1.0, value=0.574282, format="%.6f")
        spread1 = st.number_input("spread1", min_value=-7.964984, max_value=-1.0, value=-7.964984, format="%.6f")
        spread2 = st.number_input("spread2", min_value=0.006274, max_value=0.5, value=0.006274, format="%.6f")
        D2 = st.number_input("D2", min_value=1.423287, max_value=2.0, value=1.423287, format="%.6f")
        PPE = st.number_input("PPE", min_value=0.044539, max_value=1.0, value=0.044539, format="%.6f")

    # Prediction Button
    if st.button("Predict Parkinson's Disease"):
        # Create input DataFrame
        new_input_df = pd.DataFrame([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter,
                                      MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
                                      MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5,
                                      MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1,
                                      spread2, D2, PPE]], columns=columns)

        # Apply transformations
        new_input_scaled = parkinsons_scaler.transform(new_input_df)
        new_input_pca = parkinsons_pca.transform(new_input_scaled)
        confidence = parkinsons_knn_model.predict_proba(new_input_pca)[0]

        # Make prediction
        prediction = parkinsons_knn_model.predict(new_input_pca)

        # Display result
        if prediction[0] == 1:
            st.error("The model predicts that the patient has Parkinson's disease.")
        else:
            st.success("The model predicts that the patient does not have Parkinson's disease.")

        # Create DataFrame for Altair graph
        confidence_df = pd.DataFrame({
            "Prediction": ["No Parkinson's", "Parkinson's"],
            "Confidence": [confidence[0] * 100, confidence[1] * 100]  # Convert to percentage
        })

        # Create Altair bar chart
        chart = alt.Chart(confidence_df).mark_bar().encode(
            x=alt.X("Prediction", title="Prediction Outcome"),
            y=alt.Y("Confidence", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
            color="Prediction"
        ).properties(title="Prediction Confidence Level", width=500)

        # Display the chart in Streamlit
        st.altair_chart(chart, use_container_width=True)


# Lung Cancer Risk Prediction Page
if selected == "Lung Cancer Risk Prediction":
    st.title("🏥Multiple Disease Prediction System")
    st.header("🩺Lung Cancer Risk Detection")

    # User Inputs
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=14, max_value=80, value=14)
        Air_Pollution = st.slider("Air Pollution", 1, 8, 1)
        Alcohol_use = st.slider("Alcohol Use", 1, 8, 1)
        Dust_Allergy = st.slider("Dust Allergy", 1, 8, 1)
        OccuPational_Hazards = st.slider("Occupational Hazards", 1, 8, 1)
        Genetic_Risk = st.slider("Genetic Risk", 1, 7, 1)
        chronic_Lung_Disease = st.slider("Chronic Lung Disease", 1, 7, 1)
        Balanced_Diet = st.slider("Balanced Diet", 1, 7, 1)
        Obesity = st.slider("Obesity", 1, 7, 1)
        Smoking = st.slider("Smoking", 1, 8, 1)
        Passive_Smoker = st.slider("Passive Smoker", 1, 8, 1)

    with col2:
        Chest_Pain = st.slider("Chest Pain", 1, 9, 1)
        Coughing_of_Blood = st.slider("Coughing of Blood", 1, 9, 1)
        Fatigue = st.slider("Fatigue", 1, 9, 1)
        Weight_Loss = st.slider("Weight Loss", 1, 8, 1)
        Shortness_of_Breath = st.slider("Shortness of Breath", 1, 9, 1)
        Wheezing = st.slider("Wheezing", 1, 8, 1)
        Swallowing_Difficulty = st.slider("Swallowing Difficulty", 1, 8, 1)
        Clubbing_of_Finger_Nails = st.slider("Clubbing of Finger Nails", 1, 9, 1)
        Frequent_Cold = st.slider("Frequent Cold", 1, 7, 1)
        Dry_Cough = st.slider("Dry Cough", 1, 7, 1)
        Snoring = st.slider("Snoring", 1, 7, 1)

    # Gender Encoding
    gender = st.radio("Select Gender", ["Male", "Female"])
    Gender_Male = 1 if gender == "Male" else 0
    Gender_Female = 1 if gender == "Female" else 0

    # Predict Button
    if st.button("Predict Lung Cancer Stage"):
        # Create input DataFrame
        input_data = pd.DataFrame([[Age, Air_Pollution, Alcohol_use, Dust_Allergy, OccuPational_Hazards,
                                    Genetic_Risk, chronic_Lung_Disease, Balanced_Diet, Obesity, Smoking,
                                    Passive_Smoker, Chest_Pain, Coughing_of_Blood, Fatigue, Weight_Loss,
                                    Shortness_of_Breath, Wheezing, Swallowing_Difficulty, Clubbing_of_Finger_Nails,
                                    Frequent_Cold, Dry_Cough, Snoring, Gender_Male, Gender_Female]],
                                columns=lung_cancer_standard_scaler.feature_names_in_)  # Ensure feature order matches training

        # Scale the input data
        input_scaled = lung_cancer_standard_scaler.transform(input_data)

        # Make prediction
        prediction = lung_cancer_KNN_model.predict(input_scaled)
        probabilities = lung_cancer_KNN_model.predict_proba(input_scaled)[0]

        # Display result
        stage_mapping = {1: "Low", 2: "Medium", 3: "High"}
        predicted_stage = stage_mapping.get(prediction[0], "Unknown")

        if predicted_stage == "Low":
            st.success(f"The model predicts the Lung Cancer Stage as **{predicted_stage}**. (Low Risk) ✅")
        elif predicted_stage == "Medium":
            st.warning(f"The model predicts the Lung Cancer Stage as **{predicted_stage}**. (Moderate Risk) ⚠️")
        else:
            st.error(f"The model predicts the Lung Cancer Stage as **{predicted_stage}**. (High Risk) ❌")

        # Confidence Level Visualization
        confidence_df = pd.DataFrame({
            "Risk Level": ["Low", "Medium", "High"],
            "Confidence": probabilities
        })

        confidence_chart = (
            alt.Chart(confidence_df)
            .mark_bar()
            .encode(
                x=alt.X("Risk Level", sort=None),
                y="Confidence",
                color="Risk Level"
            )
            .properties(title="Prediction Confidence Level")
        )

        st.altair_chart(confidence_chart, use_container_width=True)