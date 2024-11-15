#logs.py

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

gitlog_list = load_data("../cargotracker/gitlog.json")

def print_logs(gitlog_list):
    for log in gitlog_list:
        #element = gitlog_list[log]
        print(log['message'])
        #print(clean_logs(log['message']))
        if 'stats' in log:
            print(*log['stats']['files'])

def get_logs(gitlog_list):
    final = []
    for log in gitlog_list:
        final.append(log['message'])

    return(final)

def get_stat(gitlog_list):
    final = []
    for log in gitlog_list:
        if 'stats' in log:
            #print(*log['stats']['files'])
            final.append(log['stats']['files'])   
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
    test = re.sub(r"\-", " " , text)
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

#print_logs(gitlog_list)

messages = get_logs(gitlog_list)
cleaned_mesages = clean_logs(messages)
#print(gitlog_list)
#print(messages)
#print("\n\n")
#print(cleaned_mesages)

vectorizer = TfidfVectorizer(
    lowercase=True,max_features=100,
    max_df = 0.8,
    min_df = 5,
    ngram_range = (1,3),
    stop_words = "english"
)

vectors = vectorizer.fit_transform(cleaned_mesages)
feature_names = vectorizer.get_feature_names_out()

dense = vectors.todense()
denselist = dense.tolist()

all_keywords = []

for description in denselist:
    x=0 
    keywords= []
    for word in description:
        if word > 0:
            keywords.append(feature_names[x])
        x=x+1
    all_keywords.append(keywords)
#print(cleaned_mesages[0])
#print(all_keywords)

true_k = 11


model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100,n_init=1)
model.fit(vectors)

order_centroid = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
with open ("data/messages_results.txt", "w", encoding="utf-8") as f:
    for i in range(true_k):
        f.write(f"Cluster {i}")
        f.write("\n")
        for ind in order_centroid[i, :10]:
            f.write(' %s ' % terms[ind],)
            f.write("\n")
        f.write("\n")
        f.write("\n")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



kmeans_indices = model.fit_predict(vectors)
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())
colors = [mcolors.XKCD_COLORS[color] for color in mcolors.XKCD_COLORS.keys()][:true_k]

x_axis = [o[0] for o in scatter_plot_points ]
y_axis = [o[1] for o in scatter_plot_points ]

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmeans_indices])

#for i, txt in enumerate(names): this should be the commit message
#    ax.annotate(txt[0:5], (x_axis[i], y_axis[i]))

plt.savefig("messages.png")