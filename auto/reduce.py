#reduce.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import string
from nltk.corpus import stopwords
import json
import glob
import re
from unidecode import unidecode

def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)

def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

data = load_data("../cargotracker/gitlog.json")

# List to hold the reduced JSON data
reduced_data_list = []

def create_clean(data, reduced_data_list):
# Loop through each commit entry in the list
    for entry in data: #do I want to add an if 'stats' in data?
        reduced_data = {
            
            "message": clean_message(entry.get("message")),
            "stats": {
                "files": clean_logs(entry.get("stats", {}).get("files", []))
            }
        }   
        reduced_data_list.append(reduced_data)
    return(reduced_data_list)

def clean_message(message):
    stops = stopwords.words("english")
    nonsense = load_data("data/nonsense.json")
    stops = stops+nonsense
    final = [] 
    clean_text = remove_stops(message, stops)
    final.append(clean_text)
    return(final)

def clean_logs(docs):
    stops = stopwords.words("english")
    nonsense = load_data("data/nonsense.json")
    stops = stops+nonsense
    final = []
    for doc in docs:
        clean_log = remove_stops(doc, stops)
        final.append(clean_log)
    return(final)

def remove_stops(text, stops):
    text = unidecode(text)
    text = re.sub(r"Signed-off-by:.*>","",text)
    text = re.sub(r"\.md","",text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\-", " " , text)
    text = re.sub(r"\."," ", text)
    words = text.split()
    final = []
    for word in words:
        if word not in stops:
            final.append(word)
    final = " ".join(final)
    final = final.translate(str.maketrans("","",string.punctuation))
    final = "".join([i for i in final if not i.isdigit() ])
    while "  " in final:
        final = final.replace("  ", " ")
    return(final)

# Convert the reduced data list back to JSON format 
reduced_data_list = create_clean(data, reduced_data_list)
reduced_json = json.dumps(reduced_data_list, indent=4)

# Print the reduced JSON
#print(reduced_json)
vectorizer = TfidfVectorizer(
    lowercase=True,max_features=100,
    max_df = 0.8,
    min_df = 5,
    ngram_range = (1,3),
    stop_words = "english"
)

vectors = vectorizer.fit_transform(reduced_json)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
all_keywords= []

for description in denselist:
    x=0 
    keywords= []
    for word in description:
        if word > 0:
            keywords.append(feature_names[x])
        x=x+1
    all_keywords.append(keywords)

print(all_keywords[0])