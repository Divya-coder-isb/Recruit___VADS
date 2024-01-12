#!/usr/bin/env python
# coding: utf-8

# In[42]:


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

# Define the logic for the buttons
if st.button("Apply"):
    try:
        # Prepare input data for prediction
        user_data = pd.DataFrame({
            'Role': [role],
            'Skills': [skills],
            'Experience': [experience],
            'Certification': [certification]
        })

        # Vectorize the input data
        X_pred = vectorizer.transform(user_data.astype(str).agg(' '.join, axis=1))

        # Predict the relevancy score using the trained model
        user_data['Relevancy Score'] = model.predict(X_pred)

        # Display the results
        output_table.table(user_data[["Relevancy Score"]])
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.text("Check the console or logs for more details.")
elif st.button("Clear"):
    role = ""
    skills = ""
    experience = ""
    certification = ""
    output_table.empty()

