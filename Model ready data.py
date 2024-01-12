#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import the required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import pickle

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

# Sort the DataFrame by updated_jobtitle and cosine similarity in descending order
sorted_df = df.sort_values(by=['updated_jobtitle', 'Cosine Similarity'], ascending=[True, False])

# Print the sorted DataFrame
print(sorted_df)


# In[4]:


# Save the DataFrame to a CSV file
sorted_df.to_csv("Modelreadydata.csv", header=True, index=False)


# In[5]:


# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


# In[6]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('D:\\1 ISB\\Term 2\\FP\\FP project\\Modelreadydata.csv')

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save results as CSV files
train_df.to_csv("Trainingdataset_data.csv", header=True, index=False)
test_df.to_csv("Testingdataset_data.csv", header=True, index=False)

