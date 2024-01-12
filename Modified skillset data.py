#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv (r"D:\1 ISB\Term 2\FP\FP project\Skill set data.csv")


# In[2]:


df ['Experience'] = np.random.randint (1, 31, size=len (df))


# In[3]:


certifications = ["CSPO", "PMP", "Google Project Management", "PfMP", "Agile certified", "Certified scrum master", "PgMP", "CAPM", "Six sigma", "C++", "Java", "Python", "BPA", "AIPMM", "CBAP"]
df ['Certification'] = df.apply (lambda x: ", ".join (np.random.choice (certifications, size=np.random.randint (1, 6), replace=False)), axis=1)


# In[6]:


# Rename the 'Experience' column to 'Skills Experience'
df = df.rename(columns={'Experience': 'Skills Experience'})

# Rename the 'Certification' column to 'Skills Certification'
df = df.rename(columns={'Certification': 'Skills Certification'})


# In[7]:


print (df)


# In[8]:


df.to_csv("Modifiedskillsetdata_data.csv", header=True, index=False)

