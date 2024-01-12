#!/usr/bin/env python
# coding: utf-8

# In[50]:


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

# Create the apply and clear buttons below the columns
apply_button = st.button("Apply")
clear_button = st.button("Clear")

# Display the message below the Apply button
st.markdown("Share job specifics, hit 'Apply,' and behold a dazzling lineup of ideal candidates!")

# Define the logic for the buttons
if apply_button:
    try:
        # Vectorize the user's input
        input_text = " ".join([role, skills, experience, certification])
        input_vector = vectorizer.transform([input_text])

        # Use the model to predict the relevancy score
        predicted_scores = model.predict(input_vector)

        # Add the input and predicted score to the DataFrame
        input_data = pd.DataFrame({
            "Candidate Name": ["Input"],
            "Email ID": ["Input"],
            "Relevancy Score": predicted_scores
        })

        output_df = pd.concat([input_data, data], ignore_index=True)

        # Sort the DataFrame by the relevancy score
        output_df = output_df.sort_values(by="Relevancy Score", ascending=False)

        # Convert to percentage with 2 decimal places
        output_df["Relevancy Score"] = output_df["Relevancy Score"].apply(lambda x: max(0, min(x, 1)) * 100)

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

