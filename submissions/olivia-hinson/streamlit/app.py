import streamlit as st 
import numpy as np 
import pandas as pd
import joblib 
import os


######### initial setup 
st.set_page_config(page_title='AD@streamlit', layout='wide')
st.title(':orange[Alzheimer\'s Prediction]')

@st.cache_resource
def load_model_and_preprocessor(): 
    #model_path = 'models/flight_model.pkl'
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'alzheimers_model.pkl')
    #preprocessor_path = 'preprocessor/flight_preprocessor.pkl'
    preprocessor_path = os.path.join(os.path.dirname(__file__), 'preprocessor', 'alzheimers_preprocessor.pkl')
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

def alzheimersPredict(input_data): 
    X = pd.DataFrame([input_data])
    
    # Convert 'Yes'/'No' to 1/0
    X['MemoryComplaints'] = X['MemoryComplaints'].map({'Yes': 1, 'No': 0})
    X['BehavioralProblems'] = X['BehavioralProblems'].map({'Yes': 1, 'No': 0})
    
    # Preprocess the input data
    X_processed = preprocessor.transform(X)
    
    # Grab first prediction of model
    prediction = model.predict(X_processed)[0]
    
    return prediction

######### About the app ###############
markdown_about_msg = """
        
        ## Welcome to the Alzheimer's Prediction Project
        
        This web application empowers you to analyze critical patient data that could revolutionize early Alzheimer's detection. 
        It lets you play around with key parameters to distinguish between patients with and without Alzheimer's. Currently the app
        is using a model data set (source below)
        
        Data source :  KAGGLE : [link](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) to the data set

        Field meanings: 
          - :blue[MMSE]: Mini-Mental State Examination score, ranging from 0 to 30. Lower scores indicate cognitive impairment.
          - :blue[ADL]: Activities of Daily Living score, ranging from 0 to 10. Lower scores indicate greater impairment.
          - :blue[Functional Assessment]: Functional assessment score, ranging from 0 to 10. Lower scores indicate greater impairment.
        
    """
    
############ SIDEBAR introduction to project ##########################
with st.sidebar:
    st.image(os.path.join(os.path.dirname(__file__), 'image', 'alzheimer_image.jpg'))
    st.markdown(markdown_about_msg)

col1,col2 = st.columns(2,gap="medium")

with col1: 
    mmse = st.number_input('Mini-Mental State Examination (MMSE) Rating', min_value=0, max_value=30, value='min')
    adl = st.number_input('Activities of Daily Living (ADL) Rating', min_value=0, max_value=10, value='min')
    functAsses = st.number_input('Functional Assessment Rating', min_value=0, max_value=10, value='min')

with col2: 
    memCompl = st.selectbox('Is the patient experiencing any :orange[Memory Complaints]?', ('No', 'Yes'))
    behaviorProbs = st.selectbox('Is the patient experiencing any :orange[Behavioral Problems]?', ('No', 'Yes'))

if (mmse != None and adl != None and functAsses != None and
    memCompl!= None and  behaviorProbs != None):
    
    if st.button('Generate Alzheimer Prediction'): 
        input_data = {
            'MMSE': mmse, 
            'ADL': adl,
            'FunctionalAssessment': functAsses, 
            'MemoryComplaints': memCompl, 
            'BehavioralProblems': behaviorProbs, 
        }
            
        # Make prediction
        result = alzheimersPredict(input_data)
            
        # Display result
        if (result == 1): 
            st.error('Alzheimer Diagnosis :thumbsdown:')
        else: 
            st.success('This patient doesn\'t appear to be diagnosed with Alzheimers :thumbsup:')
else: 
    alzheimerPredictButton = st.button('Generate Alzheimer Prediction', disabled=True)
             








