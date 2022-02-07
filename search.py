import pandas as pd
import numpy as np
from scipy import rand
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import nltk

import random
import pickle
import string


file_name = 'clear.csv'
index = {}
dates = {}
docs = []

vectorizer = TfidfVectorizer(lowercase=True)
lemmatizer = WordNetLemmatizer()

class Document:
    def __init__(self, title, text, year, rating):
        self.title = title
        self.text = text
        self.year = year
        self.rating = rating
    
    def format(self, query):
        return [self.title, self.text + ' ...']

def build_index():
    global index
    global docs
    global vectorizer

    with open('documents', 'rb') as f:
        docs = pickle.load(f)
    
    # Считаем и построим словарь для tf-idf
    with open('clean_data', 'rb') as file:
        data = pickle.load(file)
        vectorizer.fit_transform(data)

    # Оффлайн сортируем по рейтингу фильма 
    docs = np.array(list(filter(lambda x: x.rating > 5, docs)))

    # Строим начальный инвертированный индекс +индекс по датам выхода
    for ind, doc in enumerate(docs):
        tmp = doc.title
        if pd.notna(doc.text):
            tmp += ' ' + doc.text
        for word in set(tmp.split()):
            word = word.strip().lower()
            if word not in index.keys():
                index[word] = []
            index[word].append(ind)
        if pd.notna(doc.year):
            if doc.year not in dates.keys():
                dates[doc.year] = []
            dates[doc.year].append(ind)
        

def clean_data(sample: str):
    if pd.notna(sample):
        sample = sample.translate(str.maketrans('', '', string.punctuation))
        words = nltk.word_tokenize(sample)
        sample = ' '.join([lemmatizer.lemmatize(w) for w in words])
    return sample

def score(query, document):
    if query == '':
        return random.random()

    query_data = clean_data(query)
    document_data = clean_data(document.title + ' ' + document.text)

    query_vector = vectorizer.transform([query_data]).todense()[0]
    document_vector = vectorizer.transform([document_data]).todense()[0]
    document_vector = document_vector.reshape((query_vector.shape[1], -1))

    cos_sim = np.dot(query_vector, document_vector)/(np.linalg.norm(query_vector)*np.linalg.norm(document_vector))
    return cos_sim

def retrieve(query):
    global index
    global dates
    global docs
    
    if query == '':
        return sorted(docs, key=lambda x: -x.rating)[:5]

    keywords = query.split()
    date_docs = []
    word_docs = []
    for word in keywords:
        if word.isnumeric() is True and word in dates.keys():
            date_docs = dates[word]
        if word.lower() in index.keys():
            word_docs.append(set(index[word.lower()]))
    
    # Два указателя
    common_set = word_docs[0]
    if len(word_docs) > 1:
        for indx in word_docs[1:]:
            if len(common_set.intersection(indx)) >= 50:
                common_set = common_set.intersection(indx)
            else:
                break
    
    if len(date_docs) > 0 and len(common_set.intersection(date_docs)) >= 50:
        common_set = common_set.intersection(date_docs)

    return docs[list(common_set)][:5]
    