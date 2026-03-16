#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load required packages
import pandas as pd
import numpy as np
import re

def clean_AirBNB_data(df):
    """
    This function cleans AirBNB listings data from Inside Airbnb: http://insideairbnb.com/get-the-data/
    """
    # Drop the columns with 100% missing values 
    df = df.drop(['license','neighbourhood_group'], axis = 1)   

    # Convert datatypes for id and host_id to object
    df[['id', 'host_id']] = df[['id', 'host_id']].astype('str')

    # correct datatype mismatch for columns that should be of datetime data type
    df['last_review'] = pd.to_datetime(df['last_review'], infer_datetime_format=True)

    # fill missing reviews per month with zero
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

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


    # Rule 1: When both bathrooms and shared bathrooms are missing, impute the mean for the number of bathrooms
    mean_bathrooms = df['num_bathrooms'].mean()
    rounded_mean_bathrooms = round(mean_bathrooms)

    df['num_bathrooms'] = df.apply(lambda row: rounded_mean_bathrooms if pd.isna(row['num_bathrooms']) and pd.isna(row['shared_bath']) else row['num_bathrooms'], axis=1)
    df['shared_bath'] = df.apply(lambda row: 0 if pd.isna(row['num_bathrooms']) and pd.isna(row['shared_bath']) else row['shared_bath'], axis=1)

    # Rule 2: When bathrooms are missing and there is a value for shared bathrooms, make the bathrooms value equal to zero
    df['num_bathrooms'] = df.apply(lambda row: 0 if pd.isna(row['num_bathrooms']) and pd.notna(row['shared_bath']) else row['num_bathrooms'], axis=1)

    # Rule 3: Replace shared missing values with zero when there is a value for the number of bathrooms
    df['shared_bath'] = df.apply(lambda row: 0 if pd.notna(row['num_bathrooms']) and pd.isna(row['shared_bath']) else row['shared_bath'], axis=1)

    # fill the missing number of bedrooms with the mean or average
    df['num_bedrooms'] = df['num_bedrooms'].fillna(df['num_bedrooms'].mean())

    # Added a new has stars rating column
    df['has_star_rating'] = np.where(df['number_of_reviews'] > 2, True, False)
    
    return df

