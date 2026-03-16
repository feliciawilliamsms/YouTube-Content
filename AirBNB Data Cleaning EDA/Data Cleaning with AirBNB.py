#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning with AirBNB (Albany New York Listings)

# ## General Data Cleaning Steps:
# - 1. Drop duplicates
# - 2. Drop irrelevant data
# - 3. Fix structural erros - for example datatype mismatch
# - 4. Handling missing data - drop the feature, drop some rows, or fill in missing data (with mean, median, mode (categorical data, back or forward fill, regression, or multiple imputation methods)
# 

# ## A First Look at the Data

# In[1]:


# Load packages
import numpy as np
import pandas as pd


# Display up to 500 columns. Prevent truncation of shown columns 
pd.set_option('display.max_columns', 500)


# In[2]:


# Load the dataset
df = pd.read_csv("C:/Users/frwil/Documents/DesktopFiles/Businesses/EnhanceImpactWithData/YouTube Video Content/AB_Albany_Listings.csv")


# In[3]:


# View sample of five rows
df.sample(5)


# In[4]:


# Check the number of rows and columns of the data
df.shape


# In[5]:


# Check to see if there are duplicate rows 
len(df) - len(df.drop_duplicates())


# ## General Data Cleaning Step: Drop Duplicates
# - 1. Drop duplicates - There are no duplicates
# 

# In[6]:


# Obtain dataset information
df.info()


# In[7]:


# Calculate the percentage of missing values and sort 
round(df.isnull().sum()/len(df)*100,2).sort_values(ascending = False)


# ## General Data Cleaning Step: Drop irrelevant data
# - 2. Drop irrelevant data - licencse and neighborhood group are 100% missing and will be dropped, we will drop scrape id as well as we will not need it for analysis of listings

# In[8]:


# Drop the columns with 100% missing values
df = df.drop(['license','neighbourhood_group'], axis = 1)     
df.info()


# ## General Data Cleaning Step: Datatype mismatch and other structural errors
# - 3. Fix structural erros - for example datatype mismatch 
# - there are data type mismataches for: id (object), host_id(object), last_review (date)
# 

# In[9]:


# Convert datatypes for id and host_id to object
df[['id', 'host_id']] = df[['id', 'host_id']].astype('str')
df.info()


# In[10]:


# view a sample of three records
df.sample(3)


# In[11]:


# correct datatype mismatch for columns that should be of datetime data type
df['last_review'] = pd.to_datetime(df['last_review'], format = '%m/%d/%Y')
df.info()


# In[12]:


# Create dataframe of missing last_review and view a sample
df[df['last_review'].isnull()].sample(6)


# ## General Data Cleaning Step:  Replace missing values with  zero 
# - Replace missing values for the column reviews per month with zero  

# In[13]:


# fill missing reviews per month with zero
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# Create dataframe of missing last_review and view a sample
df[df['last_review'].isnull()].sample(6)


# In[14]:


# Calculate the percentage of missing values and sort 
round(df.isnull().sum()/len(df)*100,2).sort_values(ascending = False)


# ### Feature Engineering 
# - Using regular expressions to parse out number of bedrooms, bathrooms, and stars from the name column.

# In[15]:


# import pandas as pd
import re

# Define a function to extract values
def extract_values(text, pattern):
    match = pattern.search(text)
    return match.group(1) if match else None

# Define patterns to extract bedrooms, bathrooms, and stars
bedroom_pattern = re.compile(r'(\d+)\s*bedroom')
bathroom_pattern = re.compile(r'(\d+(\.\d+)?)\s*bath')
beds_pattern = re.compile(r'(\d+)\s*bed')
shared_bath_pattern = re.compile(r'(\d+)\s*shared')
stars_pattern = re.compile(r'★(\d+(\.\d+)?)')

# Extract and create new columns for bedrooms, bathrooms, and stars
df['num_bedrooms'] = df['name'].apply(lambda x: extract_values(x, bedroom_pattern)).astype(float)
df['num_bathrooms'] = df['name'].apply(lambda x: extract_values(x, bathroom_pattern)).astype(float)
df['num_beds'] = df['name'].apply(lambda x: extract_values(x, beds_pattern)).astype(float)
df['shared_bath'] = df['name'].apply(lambda x: extract_values(x, shared_bath_pattern)).astype(float)
df['num_stars'] = df['name'].apply(lambda x: extract_values(x, stars_pattern))  # Keeping it as a string


# In[16]:


# View a sample of the dataframe
df.sample(3)


# In[17]:


# Calculate the percentage of missing values and sort 

round(df.isnull().sum()/len(df)*100,2).sort_values(ascending = False)


# In[18]:


# View the relationship between shared_bath and num_bathrooms
df[df['num_bathrooms'].isnull()].sample(10)


# # General Data Cleaning Step: Replace missing values with zero and the mean
# - replace the shared missing values with zero when there is a value for number of bathrooms. So that one indicates shared bathroom(s) and zero indicates no shared bathrooms.
# - when there are missing values for bathrooms and missing values for shared bathrooms impute the mean for number of bathrooms.
# - when bathrooms are missing and there is a value for shared bathrooms make the bathrooms value equal to zero.
# - Leave the number of stars rating as is just look at the intersection of when stars are missing and when the last_review date is not missing.
# - use the mean to replace missing values for num_bedrooms

# In[19]:


# Rule 1: When both bathrooms and shared bathrooms are missing, impute the mean for the number of bathrooms
mean_bathrooms = df['num_bathrooms'].mean()
rounded_mean_bathrooms = round(mean_bathrooms)

df['num_bathrooms'] = df.apply(lambda row: rounded_mean_bathrooms if pd.isna(row['num_bathrooms']) and pd.isna(row['shared_bath']) else row['num_bathrooms'], axis=1)
df['shared_bath'] = df.apply(lambda row: 0 if pd.isna(row['num_bathrooms']) and pd.isna(row['shared_bath']) else row['shared_bath'], axis=1)

# Rule 2: When bathrooms are missing and there is a value for shared bathrooms, make the bathrooms value equal to zero
df['num_bathrooms'] = df.apply(lambda row: 0 if pd.isna(row['num_bathrooms']) and pd.notna(row['shared_bath']) else row['num_bathrooms'], axis=1)

# Rule 3: Replace shared missing values with zero when there is a value for the number of bathrooms
df['shared_bath'] = df.apply(lambda row: 0 if pd.notna(row['num_bathrooms']) and pd.isna(row['shared_bath']) else row['shared_bath'], axis=1)


# In[20]:


# Calculate the percentage of missing values and sort 

round(df.isnull().sum()/len(df)*100,2).sort_values(ascending = False)


# In[21]:


# fill the missing number of bedrooms with the mean or average
df['num_bedrooms'] = df['num_bedrooms'].fillna(df['num_bedrooms'].mean())
round(df.isnull().sum()/len(df)*100,2).sort_values(ascending = False)


# ###  Determine how to treat missing values by looking at the relationships between missing and non-missing fields

# In[22]:


# viewing and calculating the number of reviews for listings with/without reviews
no_stars = df[df['num_stars'].isnull()]
no_stars.sample(6)


# In[23]:


# find the distribution of number of reviews when there are no stars
no_stars['number_of_reviews'].describe()


# In[24]:


has_stars = df[~df['num_stars'].isnull()]
has_stars.sample(6)


# In[25]:


# find the distribution of number of reviews when there are stars
has_stars['number_of_reviews'].describe()


# ### Making the decision to not impute some of the missing fields without further analysis or EDA.
# - The missing star ratings and last_review date will not have there null values replaced.

# In[ ]:





# ### Feature Engineering 
# - Adding an new has_star_rating column.

# In[26]:


df['has_star_rating'] = np.where(df['number_of_reviews']>2, True, False)
df.sample(5)


# In[27]:


df.sample(20)


# In[ ]:




