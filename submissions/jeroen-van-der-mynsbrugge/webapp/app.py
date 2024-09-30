import streamlit as st
from catboost import CatBoostClassifier
import numpy as np

# Load the trained CatBoost model
model = CatBoostClassifier()
model.load_model('alzheimers_model_optimized.cbm')

# App layout
st.image("header.png")

# Description
st.write("""
### About this App 
This app predicts whether a patient is likely to be diagnosed with Alzheimer's Disease based on key clinical features. 
It uses a binary classification model, trained on a publicly available dataset [1] containing detailed health information 
and Alzheimer's Disease diagnoses for 2,149 patients.


##### Model Details
The prediction model is a CatBoost classifier. Initially, it was trained on all 32 features in the dataset, including 
demographic information, lifestyle factors, medical history, clinical measurements, symptoms, and cognitive and 
functional assessments. 
\n
After performing feature importance analysis, the following 5 features were identified as the most predictive:

- **Functional Assessment Score (FA)**: Between 0 and 10. Lower scores indicate greater impairment.
- **Activities of Daily Living (ADL) Score**: Between 0 and 10. Lower scores indicate greater impairment.
- **Mini-Mental State Examination (MMSE) Score**: Between 0 and 30. Lower scores indicate cognitive impairment.
- **Memory Complaints**: Indicates if the patient reports memory issues (Yes/No)
- **Behavioral Problems**: Indicates if the patient has behavioral issues (Yes/No)

The model was retrained using only these 5 key features and fine-tuned to match the performance of the original model 
with all 32 features, achieving a mean accuracy of 95.57%, validated through k-fold cross-validation.

##### Dataset Citation
1. Rabie El Kharoua, _Alzheimer's Disease Dataset_, Kaggle, 2024, https://doi.org/10.34740/KAGGLE/DSV/8668279.

""")

# Get user input
st.write("""
---
### Enter Patient Data
""")
fa = st.text_input("Functional Assessment Score (0-10)", value="0.0")
adl = st.text_input("Activities of Daily Living (ADL) Score (0-10)", value="0.0")
mmse = st.text_input("Mini-Mental State Exam (MMSE) Score (0-30)", value="0.0")

memory_complaints = st.radio("Memory Complaints", ('No', 'Yes'))
behavioral_problems = st.radio("Behavioral Problems", ('No', 'Yes'))

# Encode categorical variables
mc = 1 if memory_complaints == 'Yes' else 0
bp = 1 if behavioral_problems == 'Yes' else 0

# Run prediction model and show result
if st.button('Run Prediction Model'):
    try:
        fa = float(fa)
        adl = float(adl)
        mmse = float(mmse)

        # Check if the values are within the predefined ranges
        if not (0 <= fa <= 10):
            st.error("Functional Assessment Score must be between 0 and 10.")
        elif not (0 <= adl <= 10):
            st.error("ADL Score must be between 0 and 10.")
        elif not (0 <= mmse <= 30):
            st.error("MMSE Score must be between 0 and 30.")
        else:
            input_data = np.array([[fa, adl, mmse, mc, bp]])

            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.error("Positive Alzheimer's diagnosis likely.")
            else:
                st.success("Negative Alzheimer's diagnosis likely.")

    except ValueError:
        st.error("Please ensure that all inputs are numeric.")
st.write("""
---
""")
