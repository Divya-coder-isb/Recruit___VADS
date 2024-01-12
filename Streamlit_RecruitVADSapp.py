#!/usr/bin/env python
# coding: utf-8

# In[30]:


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

# Define a function to calculate the cosine similarity
def get_cosine_similarity(user_text, candidate_text):
    user_vector = vectorizer.transform([user_text])
    candidate_vector = vectorizer.transform([candidate_text])
    similarity = cosine_similarity(user_vector, candidate_vector)[0][0]
    return similarity

# Define the logic for the buttons
if st.button("Apply"):
    try:
        # Concatenate the user's text
        user_text = " ".join([role, skills, experience, certification])
        # Apply the function to the DataFrame
        data['Cosine Similarity'] = data.apply(lambda row: get_cosine_similarity(user_text, " ".join([str(row["Role"]), str(row["Skills"]), str(row["Experience"]), str(row["Certification"])])), axis=1)
        # Extract the relevant columns for prediction
        prediction_features = data[['sorted_skills', 'Certification', 'Experience', 'Cosine Similarity']]
        X_pred = vectorizer.transform(prediction_features.astype(str).agg(' '.join, axis=1))
        # Predict the relevancy score using the trained model
        data['Relevancy Score'] = model.predict(X_pred)
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

