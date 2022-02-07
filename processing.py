"""
Preprocess initial dataframe with data, using multiprocessing and 
produce a pickle file with Document class objects.

Launching only once.
"""

import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')

import multiprocessing as mp
from processing import *
import re
from search import Document
import pickle
import string


lemmatizer = WordNetLemmatizer()
file_name = 'movie_data_1950_2020.csv'
useful_columns = ['query', 'description', 'IMDb_rating', 'rotten_tomatoes_rating', 'meta_critic_rating']

def clear_rating(sample):
    if pd.isna(sample):
        return sample
    
    sep = None
    if '/' in sample:
        sep = '/'    
    elif '%' in sample:
        sep = '%'
    if sep is not None:
        return float(sample.split(sep)[0])
    else:
        return np.nan

def delete_extra_symbols(sample):
    return re.sub(r'["!]', r"", sample)

def title_and_year(sample):
    if 'movie' in sample and len(sample.split(' movie ')) > 1:
        title_year = sample.split(' movie ')
        return [title_year[0], title_year[1]]
    elif 'movie' in sample and len(sample.split(' movie')) < 2:
        return np.nan, np.nan
    else:
        return [sample, np.nan]

def create_documents(sample):
    return Document(sample[1]['Title'], 
                    sample[1]['description'], 
                    sample[1]['Year'], 
                    sample[1]['Rating'])

def clean_data(sample: str):
    if pd.notna(sample):
        sample = sample.translate(str.maketrans('', '', string.punctuation))
        words = nltk.word_tokenize(sample)
        sample = ' '.join([lemmatizer.lemmatize(w) for w in words])
    return sample


if __name__ == "__main__":
    pool = mp.Pool(8)
    
    df = pd.read_csv(file_name, usecols=useful_columns)
    ratings = ['IMDb_rating', 'rotten_tomatoes_rating', 'meta_critic_rating']
    for source in ratings:
        df[source] = pool.map(clear_rating, df[source].tolist())

    df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'] / 10
    df['meta_critic_rating'] = df['meta_critic_rating'] / 10
    df['Rating'] = df[ratings].mean(axis=1)

    df['query'] = pool.map(delete_extra_symbols, df['query'].tolist())
    test_df = pd.DataFrame(pool.map(title_and_year, df['query'].tolist()), columns=['Title', 'Year'])
    df = pd.concat([df, test_df], axis=1)
    df = df[['Title', 'description', 'Year', 'Rating']]

    df.dropna(subset=['Title'], inplace=True)
    
    df['description'].fillna('', inplace=True)
    data = df['Title'] + ' ' + df['description']
    data = pool.map(clean_data, data.tolist())

    with open('clean_data', 'wb') as f:
        pickle.dump(data, f)
    
    docs = pool.map(create_documents, df.iterrows())
    with open('documents', 'wb') as f:
        pickle.dump(docs, f)
