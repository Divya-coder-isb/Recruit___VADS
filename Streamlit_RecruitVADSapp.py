#!/usr/bin/env python
# coding: utf-8

# In[5]:


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

# Define a function to calculate the relevancy score
def get_relevancy_score(job_title, skills, experience, certification):
    input_text = " ".join([job_title, skills, experience, certification])
    input_vector = vectorizer.transform([input_text])
    score = model.predict(input_vector)[0]
    return score

# Set the title of the app
st.title("Recruit VADS")

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
    data["Relevancy Score"] = data.apply(lambda row: get_relevancy_score(job_title, skills, experience, certification), axis=1)
    data["Relevancy Score"] = data["Relevancy Score"].apply(lambda x: "{:.2f}%".format(x*100))  # Convert to percentage with 2 decimal places
    data = data.sort_values(by="Relevancy Score", ascending=False)
    start_index = st.number_input("Start index", min_value=0, max_value=len(data)-1, value=0)
    end_index = st.number_input("End index", min_value=0, max_value=len(data)-1, value=10)
    output_table.table(data[["Candidate Name", "Email ID", "Relevancy Score"]].iloc[start_index:end_index])
elif clear_button:
    job_title = ""
    skills = ""
    experience = ""
    certification = ""
    output_table.empty()

