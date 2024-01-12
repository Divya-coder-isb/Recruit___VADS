#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Import the required libraries
import streamlit as st
import pandas as pd
import pickle
import requests

# Load the data and the model from the given paths
data_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Modifiedresumedata_data.csv"
image_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/RecruitVADSlogo.jpg?raw=true"
model_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Recruit_VADS_model.pkl?raw=true"
vectorizer_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Tfidf_Vectorizer.pkl?raw=true"

data = pd.read_csv(data_url)
model = pickle.loads(requests.get(model_url).content)
vectorizer = pickle.loads(requests.get(vectorizer_url).content)

# Define a function that takes the input from the UI and returns the relevancy score
def get_relevancy_score(job_title, skills, certification, experience):
    # Create a vector from the input
    input_features = [job_title, skills, certification, experience]
    input_vector = vectorizer.transform(input_features).toarray()
    
    # Compute the cosine similarity with the model
    similarity = model.dot(input_vector.T)
    
    # Sort the candidates by descending order of similarity
    sorted_indices = similarity.argsort(axis=0)[::-1]
    sorted_similarity = similarity[sorted_indices]
    
    # Format the output as a dataframe with candidate name, email and relevancy score
    output = pd.DataFrame()
    output['Candidate Name'] = data['Candidate Name'][sorted_indices].squeeze()
    output['Email ID'] = data['Email ID'][sorted_indices].squeeze()
    output['Relevancy Score'] = (sorted_similarity * 100).round(2).squeeze()
    output['Relevancy Score'] = output['Relevancy Score'].astype(str) + '%'
    
    return output

# Display the image on top of the page
st.image(image_url, use_column_width=True)

# Create a two-column layout for the input and output fields
col1, col2 = st.columns(2)

# Create the input fields in the left column
with col1:
    st.header("Input Fields")
    job_title = st.text_input("Job Title")
    skills = st.text_input("Skills")
    experience = st.text_input("Experience")
    certification = st.text_input("Certification")

# Create the output field in the right column
with col2:
    st.header("Output Field")
    output_table = st.empty()

# Create the apply and clear buttons below the columns
apply_button = st.button("Apply")
clear_button = st.button("Clear")

# Define the logic for the buttons
if apply_button:
    data["Relevancy Score"] = data.apply(get_relevancy_score, axis=1)
    data["Relevancy Score"] = data["Relevancy Score"].apply(lambda x: "{:.2f}%".format(x*100))  # Convert to percentage with 2 decimal places
    data = data.sort_values(by="Relevancy Score", ascending=False)
    # Display all the records
    output_table.table(data[["Candidate Name", "Email ID", "Relevancy Score"]])
elif clear_button:
    job_title = ""
    skills = ""
    experience = ""
    certification = ""
    output_table.empty()

