#!/usr/bin/env python
# coding: utf-8

# In[24]:


# Import the required libraries
import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Load the data and the model from the given paths
data_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Modifiedresumedata_data.csv"
image_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/RecruitVADSlogo.jpg?raw=true"
model_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Recruit_VADS_model.pkl?raw=true"
vectorizer_url = "https://raw.githubusercontent.com/Divya-coder-isb/Recruit___VADS/main/Tfidf_Vectorizer.pkl?raw=true"

data = pd.read_csv(data_url)
model = pickle.loads(requests.get(model_url).content)
vectorizer = pickle.loads(requests.get(vectorizer_url).content)

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

# Create the input fields in the left column
role = col1.text_input("Role")
skills = col1.text_input("Skills")
experience = col1.text_input("Experience")
certification = col1.text_input("Certification")

# Create the output field in the right column
output_table = col2.empty()

# Define user_vector globally
user_vector = None

# Define a function to calculate the relevancy score
def get_relevancy_score(row):
    global user_vector  # Access the global variable
    # Extract the candidate's information from the row
    job_title = str(row["Role"]) if pd.notnull(row["Role"]) else ""
    skills = str(row["Skills"]) if pd.notnull(row["Skills"]) else ""
    experience = str(row["Experience"]) if pd.notnull(row["Experience"]) else ""
    certification = str(row["Certification"]) if pd.notnull(row["Certification"]) else ""
    # Concatenate the candidate's text
    candidate_text = " ".join([job_title, skills, experience, certification])
    # Vectorize the candidate's text
    candidate_vector = vectorizer.transform([candidate_text])
    # Calculate the cosine similarity between the user's vector and the candidate's vector
    similarity = cosine_similarity(user_vector, candidate_vector)[0][0]
    # Return the similarity score
    return similarity

# Define the logic for the buttons
if st.button("Apply"):
    try:
        # Concatenate the user's text
        user_text = " ".join([role, skills, experience, certification])
        # Vectorize the user's input
        user_vector = vectorizer.transform([user_text])
        # Apply the function to the DataFrame
        data['Relevancy Score'] = data.apply(get_relevancy_score, axis=1)
        # Predict the relevancy score using the trained model
        data['Relevancy Score'] = model.predict(data['Relevancy Score'].values.reshape(-1, 1))
        # Sort the DataFrame by the relevancy score
        output_df = data.sort_values(by="Relevancy Score", ascending=False)
        # Convert to percentage with 2 decimal places
        output_df["Relevancy Score"] = output_df["Relevancy Score"].apply(lambda x: "{:.2f}%".format(x*100))
        # Display all the records
        output_table.table(output_df[["Candidate Name", "Email ID", "Relevancy Score"]].reset_index(drop=True))
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.text("Check the console or logs for more details.")
elif st.button("Clear"):
    role = ""
    skills = ""
    experience = ""
    certification = ""
    output_table.empty()

