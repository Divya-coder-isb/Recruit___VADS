#!/usr/bin/env python
# coding: utf-8

# In[71]:


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

from sklearn.metrics.pairwise import cosine_similarity

# Define a function to calculate the relevancy score
def get_relevancy_score(row):
    # Extract the candidate's information from the row
    job_title = str(row["Role"]) if pd.notnull(row["Role"]) else ""
    skills = str(row["Skills"]) if pd.notnull(row["Skills"]) else ""
    experience = str(row["Experience"]) if pd.notnull(row["Experience"]) else ""
    certification = str(row["Certification"]) if pd.notnull(row["Certification"]) else ""
    # Concatenate the candidate's text
    candidate_text = " ".join([job_title, skills, experience, certification])
    # Vectorize the candidate's text and the user's input
    candidate_vector = vectorizer.transform([candidate_text])
    input_vector = vectorizer.transform([" ".join([role, skills, experience, certification])])
    # Calculate the cosine similarity between the two vectors
    score = cosine_similarity(candidate_vector, input_vector)[0][0]
    # Return the score
    return score

# Display the image on top of the page with increased width
image_width = 900  # Adjust the width according to your preference
st.markdown(
    f'<img src="{image_url}" alt="image" style="width:{image_width}px;height:auto;">',
    unsafe_allow_html=True
)

# Create a two-column layout for the input and output fields
col1, col2 = st.columns(2)

# Adjust the width of each column
col1_width = image_width // 2
col2_width = image_width // 2
col1.width = col1_width
col2.width = col2_width

# Set the background style for the input container in the left column
input_container_style = """
    <style>
        div.stTextInput, div.stTextArea {
            background-color: rgb(221, 221, 221);
            border: 1px dotted black;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
    </style>
"""
st.markdown(input_container_style, unsafe_allow_html=True)

# Create the input fields within a container in the left column
with col1:
    with st.container():
        role = st.text_input("Role")
        skills = st.text_input("Skills")
        experience = st.text_input("Experience")
        certification = st.text_input("Certification")

# Create the output field in the right column
output_container = col2.container()

# Create the apply and clear buttons below the columns
apply_button = st.button("Apply", key="apply_button")
clear_button = st.button("Clear")

# Display the message below the Apply button
st.markdown("<p style='color: grey; font-style: italic;'>Share job specifics, hit 'Apply,' and behold a dazzling lineup of ideal candidates!</p>", unsafe_allow_html=True)

# Define the logic for the buttons
if apply_button:
    try:
        # Apply the function to the DataFrame
        data['Relevancy Score'] = data.apply(get_relevancy_score, axis=1)
        # Sort the DataFrame by the relevancy score
        output_df = data.sort_values(by="Relevancy Score", ascending=False)
        # Convert to percentage with 2 decimal places
        output_df["Relevancy Score"] = output_df["Relevancy Score"].apply(lambda x: "{:.2f}%".format(x*100))
        # Display all the records in a custom scrollable container
        with output_container:
            st.table(output_df[["Candidate Name", "Email ID", "Relevancy Score"]].reset_index(drop=True))
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.text("Check the console or logs for more details.")
elif clear_button:
    role = ""
    skills = ""
    experience = ""
    certification = ""
    output_container.empty()

