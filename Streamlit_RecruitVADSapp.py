#!/usr/bin/env python
# coding: utf-8

# In[29]:


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
def get_relevancy_score(input_text):
    # Vectorize the input text using the vectorizer
    input_vector = vectorizer.transform([input_text])
    # Predict the relevancy score using the model
    score = model.predict(input_vector)[0]
    # Return the score
    return score

# Display the image on top of the page
st.image(image_url, use_column_width=True)

# Create a two-column layout for the input and output fields
col1, col2 = st.columns(2)

# Create the input fields in the left column
with col1:
    st.header("Input Fields")
    role = st.text_input("Role")
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
    try:
        # Concatenate the input fields into a single string
        input_text = " ".join([role, skills, experience, certification])
        # Apply the function to the DataFrame
        data['Relevancy Score'] = data.apply(lambda row: get_relevancy_score(input_text), axis=1)
        # Convert to percentage with 2 decimal places
        data["Relevancy Score"] = data["Relevancy Score"].apply(lambda x: "{:.2f}%".format(x*100))  
        # Sort the DataFrame by the relevancy score
        output_df = data.sort_values(by="Relevancy Score", ascending=False)
        # Display all the records
        output_table.table(output_df[["Candidate Name", "Email ID", "Relevancy Score"]])
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.text("Check the console or logs for more details.")
elif clear_button:
    role = ""
    skills = ""
    experience = ""
    certification = ""
    output_table.empty()

