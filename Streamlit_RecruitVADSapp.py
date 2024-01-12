#!/usr/bin/env python
# coding: utf-8

# In[21]:


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
def get_relevancy_score(role, skills, experience, certification):
    print("Debug: Inside get_relevancy_score function")
    print(f"Debug: Input values - Role: {role}, Skills: {skills}, Experience: {experience}, Certification: {certification}")

    # Convert input values to strings
    role = str(role)
    skills = str(skills)
    experience = str(experience)
    certification = str(certification)

    # Create a vector from the input
    input_features = [role, skills, experience, certification]
    input_vector = vectorizer.transform(input_features).toarray()
    
    # Compute the cosine similarity with the model
    similarity = model.dot(input_vector.T)
    
    # Sort the candidates by descending order of similarity
    sorted_indices = similarity.argsort(axis=0)[::-1]
    sorted_similarity = similarity[sorted_indices]
    
    # Format the output as a dataframe with candidate name, email, and relevancy score
    output = pd.DataFrame()
    output['Candidate Name'] = data['Candidate Name'][sorted_indices].squeeze()
    output['Email ID'] = data['Email ID'][sorted_indices].squeeze()
    output['Relevancy Score'] = (sorted_similarity * 100).round(2).squeeze()
    output['Relevancy Score'] = output['Relevancy Score'].astype(str) + '%'
    
    print("Debug: Output DataFrame:")
    print(output)
    return output

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
        # Use lambda function to unpack the row into individual arguments
        data["Relevancy Score"] = data.apply(lambda row: get_relevancy_score(row['Role'], row['Skills'], row['Experience'], row['Certification']), axis=1)
        data["Relevancy Score"] = data["Relevancy Score"].apply(lambda x: "{:.2f}%".format(x*100))  # Convert to percentage with 2 decimal places
        data = data.sort_values(by="Relevancy Score", ascending=False)
        # Display all the records
        output_table.table(data[["Candidate Name", "Email ID", "Relevancy Score"]])
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.text("Check the console or logs for more details.")
elif clear_button:
    role = ""
    skills = ""
    experience = ""
    certification = ""
    output_table.empty()

