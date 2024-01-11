#!/usr/bin/env python
# coding: utf-8

# In[11]:


import streamlit as st
import pandas as pd
import pickle
from PIL import Image
import requests

# Load your data
df = pd.read_csv('https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Modifiedresumedata_data.csv')

# Download and load the model
model_url = 'https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Recruit_VADS_model.pkl'
model_file = requests.get(model_url)
model = pickle.loads(model_file.content)

# Download and load the vectorizer
vectorizer_url = 'https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Tfidf_Vectorizer.pkl'
vectorizer_file = requests.get(vectorizer_url)
vectorizer = pickle.loads(vectorizer_file.content)

# Load your images
image = Image.open(requests.get('https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/pngtree-online-remote-recruitment-png-image_5413767.jpg', stream=True).raw)
logo = Image.open(requests.get('https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Recruit%20VADS%20logo.png', stream=True).raw)

# Display the image and logo
st.image(image, use_column_width=True)
st.image(logo, use_column_width=True)

# Create your input fields
job_title = st.sidebar.text_input('Job Title')
skills = st.sidebar.text_input('Skills')
experience = st.sidebar.text_input('Experience')
certification = st.sidebar.text_input('Certification')

# Define a function that takes the input from the UI and returns the relevancy score
def get_relevancy_score(job_title, skills, certification, experience):
    # Create a vector from the input
    input_features = [job_title, skills, certification, experience]
    input_vector = vectorizer.transform(input_features).toarray()

    # Predict the scores using the model
    scores = model.predict(input_vector)

    # Sort the candidates by descending order of scores
    sorted_indices = scores.argsort()[::-1]
    sorted_scores = scores[sorted_indices]

    # Format the output as a dataframe with candidate name, email, and relevancy score
    output = pd.DataFrame()
    output['Candidate Name'] = df['Candidate Name'][sorted_indices].squeeze()
    output['Email ID'] = df['Email ID'][sorted_indices].squeeze()
    output['Relevancy Score'] = (sorted_scores * 100).round(2).squeeze()
    output['Relevancy Score'] = output['Relevancy Score'].apply(lambda x: f'{x:.2f}%')

    return output

# Create your buttons
if st.sidebar.button('Apply'):
    # Process the inputs and run your model
    output_df = get_relevancy_score(job_title, skills, certification, experience)

    # Display the output table
    st.table(output_df)

if st.sidebar.button('Clear'):
    # Clear the input fields
    job_title = ''
    skills = ''
    experience = ''
    certification = ''

