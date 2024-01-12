#!/usr/bin/env python
# coding: utf-8

# In[56]:


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

# Apply custom CSS styling for the input fields
st.markdown(
    """
    <style>
        .css-1q0z6kh {
            background: linear-gradient(to right, #f0f0f0, #d0d0d0);
            border-radius: 10px;
            padding: 8px;
            box-shadow: 2px 2px 5px #888888;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Create a container for the input fields in the left column with styling
with st.container():
    st.markdown('<div class="css-1q0z6kh">Role:</div>', unsafe_allow_html=True)
    role = st.text_input("Enter the desired role", value=role)
    
    st.markdown('<div class="css-1q0z6kh">Skills:</div>', unsafe_allow_html=True)
    skills = st.text_input("Enter the relevant skills", value=skills)
    
    st.markdown('<div class="css-1q0z6kh">Experience:</div>', unsafe_allow_html=True)
    experience = st.text_input("Enter the required experience", value=experience)
    
    st.markdown('<div class="css-1q0z6kh">Certification:</div>', unsafe_allow_html=True)
    certification = st.text_input("Enter the relevant certification", value=certification)

# Create the output field in the right column
output_table = st.empty()

# Create the apply and clear buttons below the columns
apply_button = st.button("Apply")
clear_button = st.button("Clear")

# Display the message below the Apply button
st.markdown("Share job specifics, hit 'Apply,' and behold a dazzling lineup of ideal candidates!")

# Define the logic for the buttons
if apply_button:
    try:
        # Apply the function to the DataFrame
        data['Relevancy Score'] = data.apply(get_relevancy_score, axis=1)
        # Sort the DataFrame by the relevancy score
        output_df = data.sort_values(by="Relevancy Score", ascending=False)
        # Convert to percentage with 2 decimal places
        output_df["Relevancy Score"] = output_df["Relevancy Score"].apply(lambda x: "{:.2f}%".format(x*100))
        # Display all the records
        output_table.table(output_df[["Candidate Name", "Email ID", "Relevancy Score"]].reset_index(drop=True))
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.text("Check the console or logs for more details.")
elif clear_button:
    role = ""
    skills = ""
    experience = ""
    certification = ""
    output_table.empty()

