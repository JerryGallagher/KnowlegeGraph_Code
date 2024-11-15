#lda.py

import pandas as pd
import numpy as np
import gensim
import gensim.corpora as  corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from nltk.corpus import stopwords
import json
import glob
import re
from unidecode import unidecode

stop_words = stopwords.words('english')
def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)

def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

#data = load_data("../cargotracker/gitlog.json")

# List to hold the reduced JSON data
reduced_data_list = []

df = pd.read_json("../cargotracker/gitlog.json")
data = df.message.values.tolist()

