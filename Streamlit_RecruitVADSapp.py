#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import the required libraries
import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define a function to load the data and the model
def load_data_and_model(data_url):
    # Load the data and the model from the given paths
    image_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/RecruitVADSlogo.jpg?raw=true"
    model_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Recruit_VADS_model.pkl?raw=true"
    vectorizer_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Tfidf_Vectorizer.pkl?raw=true"

    data = pd.read_csv(data_url)
    model = pickle.loads(requests.get(model_url).content)
    vectorizer = pickle.loads(requests.get(vectorizer_url).content)
    image = requests.get(image_url).content
    return data, model, vectorizer, image

# Modified Resume Data URL
modified_resume_data_url = "https://github.com/Divya-coder-isb/Recruit___VADS/raw/main/Modifiedresumedata_data.csv"

# Create a sidebar with an input field for the Modified Resume Data URL
st.sidebar.header("Modified Resume Data Path")
data, model, vectorizer, image = load_data_and_model(modified_resume_data_url)

# Define a function to calculate the relevancy score
def get_relevancy_score(row):
    # Extract the candidate's information from the row
    job_title = str(row["Role"]) if pd.notnull(row["Role"]) else ""
    skills = str(row["Skills"]) if pd.notnull(row["Skills"]) else ""
    experience = str(row["Experience"]) if pd.notnull(row["Experience"]) else ""
    certification = str(row["Certification"]) if pd.notnull(row["Certification"]) else ""
    # Concatenate the candidate's text
    candidate_text = " ".join([job_title, skills, experience, certification])
    # Vectorize the candidate's text using the Modified Resume Data
    candidate_vector = vectorizer.transform([candidate_text])
    
    # Vectorize the user's input
    input_text = " ".join([role, skills, experience, certification])
    input_vector = vectorizer.transform([input_text])

    # Calculate the cosine similarity between the candidate and user vectors
    cosine_sim = cosine_similarity(candidate_vector, input_vector)
    
    # Use the trained model to predict the relevancy score
    relevancy_score = model.predict(cosine_sim)[0]
    
    # Clip the score to the range [0, 100]
    relevancy_score = np.clip(relevancy_score, 0, 100)
    
    # Return the score
    return relevancy_score

# Display the image on top of the page using st.image
st.image(image, use_column_width=True)

# Create a two-column layout for the input and output fields
col1, col2 = st.columns([1, 1])

# Create the input fields in the left column
role = col1.text_input("Role")
skills = col1.text_input("Skills")
experience = col1.text_input("Experience")
certification = col1.text_input("Certification")

# Print the user's input
st.write("Role:", role)
st.write("Skills:", skills)
st.write("Experience:", experience)
st.write("Certification:", certification)

# Create the output field in the right column
output_table = col2.empty()

# Create the apply and clear buttons below the columns
apply_button = st.button("Apply")
clear_button = st.button("Clear")

# Display the message below the Apply button using st.success
st.success("Share job specifics, hit 'Apply,' and behold a dazzling lineup of ideal candidates!")

# Define the logic for the buttons
if apply_button:
    try:
        # Apply the function to the DataFrame
        data['Relevancy Score'] = data.apply(get_relevancy_score, axis=1)
        # Sort the DataFrame by the relevancy score
        output_df = data.sort_values(by="Relevancy Score", ascending=False)
        # Display all the records
        output_table.table(output_df[["Candidate Name", "Email ID", "Relevancy Score"]].reset_index(drop=True))
    except Exception as e:
        # Display the exception using st.exception and stop the execution using st.stop
        st.exception(e)
        st.stop()
elif clear_button:
    role = ""
    skills = ""
    experience = ""
    certification = ""
    output_table.empty()

