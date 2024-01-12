#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install --upgrade scikit-learn


# In[3]:


get_ipython().system('pip freeze > requirements.txt')


# In[4]:


# 1. Model training


# In[22]:


# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import accuracy_score, precision_score

# Load the data
df = pd.read_csv('D:\\1 ISB\\Term 2\\FP\\FP project\\Mergeddataset.csv')

# Combine the text of both sets of columns
combined_text = pd.concat([
    df[['updated_jobtitle', 'sorted_skills', 'Skills Experience', 'Skills Certification']].apply(lambda x: ' '.join(x.astype(str)), axis=1),
    df[['Role', 'Skills', 'Certification', 'Experience']].apply(lambda x: ' '.join(x.astype(str)), axis=1)
])

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer().fit(combined_text)

# Transform each set of columns separately
X1 = vectorizer.transform(df[['updated_jobtitle', 'sorted_skills', 'Skills Experience', 'Skills Certification']].apply(lambda x: ' '.join(x.astype(str)), axis=1))
X2 = vectorizer.transform(df[['Role', 'Skills', 'Certification', 'Experience']].apply(lambda x: ' '.join(x.astype(str)), axis=1))

# Compute the cosine similarity for each row and store the result in the DataFrame
df['Cosine Similarity'] = [cosine_similarity(X1[i], X2[i]).flatten()[0] for i in range(X1.shape[0])]

# Convert the cosine similarity to percentage and round to 2 decimal places
df['Cosine Similarity'] = (df['Cosine Similarity'] * 100).round(2)

# Extract relevant columns for training
train_features = df[['sorted_skills', 'Certification', 'Experience', 'Cosine Similarity']]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_features.astype(str).agg(' '.join, axis=1))

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, df['Cosine Similarity'])

# Save the model using pickle
model_filename = 'Recruit_VADS_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer
vectorizer_filename = 'Tfidf_Vectorizer.pkl'
with open(vectorizer_filename, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model trained and saved as", model_filename)
print("Vectorizer saved as", vectorizer_filename)


# In[23]:


from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Define a threshold for the relevancy score
threshold = 0.5

# Convert the predicted and actual relevancy scores to binary classes
y_pred_binary = [1 if x >= threshold else 0 for x in y_pred]
y_true_binary = [1 if x >= threshold else 0 for x in test_data['Relevancy Score (%)']]

# Calculate the accuracy and precision
accuracy = accuracy_score(y_true_binary, y_pred_binary) * 100
precision = precision_score(y_true_binary, y_pred_binary) * 100

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")

# Calculate the confusion matrix
confusion = confusion_matrix(y_true_binary, y_pred_binary)
print(f"Confusion Matrix: \n{confusion}")


# In[15]:


# Model load and save


# In[25]:


import pickle

# Save the model and vectorizer
model_filename = 'Recruit_VADS_model.pkl'
vectorizer_filename = 'Tfidf_Vectorizer.pkl'

with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(vectorizer_filename, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Load the model and vectorizer
loaded_model = pickle.load(open(model_filename, 'rb'))
loaded_vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

