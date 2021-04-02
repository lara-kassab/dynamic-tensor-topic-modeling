import os
import pickle
import time
import gensim
from gensim import corpora,models
import datetime
import random
import collections
import numpy as np
from gensim import corpora,models
from covid19.text import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

from config import data_dir, results_dir
from covid19 import plotting, utils

rank = 20
num_keywords = 5
num_time_slices = 90
data_name = "top_1000_daily_rank20"  # or "random_1000_daily_rank20"
local_path = results_dir
directory = os.path.join(
            data_dir,
            "datatweets_from_2020-02-01-00_to_2020-05-01-00.pickle",
        ) # Fill
save_figures = True
load_lda_model = False # True for loading trained LDA model; False for training a model
random.seed(1)

#Load corpus
corpus = pickle.load(open(directory, "rb"))

#TFIDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features= 5000,
    ngram_range=(1, 2),
    stop_words=stopwords.stopwords
)
vectors = vectorizer.fit_transform(corpus)
features = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = np.array(dense).transpose()

## LDA Preprocessing
print("Data Preprocessing...")
# Data Preprocessing
tokenized_docs = list(map(vectorizer.build_analyzer(),corpus)) # pre-process and tokenize documents using tfidfvectorizer

# Filter processed_docs based on features list
processed_docs = []
for doc in tokenized_docs:
    processed_docs.append([wd_str for wd_str in doc if(wd_str in features)])
dictionary = gensim.corpora.Dictionary(processed_docs) # define a (gensim) dictionary

# Create bag-of-words corpus (gensim format)
corpus=[dictionary.doc2bow(doc) for doc in processed_docs]

# Check if dictionaries are the same
#print(collections.Counter(list(dictionary.values())) == collections.Counter(features))

## Run LDA
print("Running LDA...")

run_LDA_tfidf = False

if run_LDA_tfidf:
    # Convert BOW Corpus to TFIDF Corpus
    tfidf=models.TfidfModel(corpus)
    corpus=tfidf[corpus]

if load_lda_model:
    lda_model = pickle.load(open(os.path.join(local_path,"lda_model_Twitter.pickle"), "rb"))
else:
    lda_model = gensim.models.LdaModel(corpus,
                                     num_topics=rank,
                                     id2word = dictionary,
                                     passes = 5,
                                     minimum_probability = 0,
                                     random_state = 1
                                      )

    with open(os.path.join(local_path,"lda_model_Twitter.pickle"), "wb") as f:
        pickle.dump(lda_model, f)

# LDA representation matrix (topics by documents)
lda_rep_mat = np.empty((rank, len(corpus)))
for numb in range(len(corpus)):
    for index, score in lda_model.get_document_topics(corpus[numb], minimum_probability = 0):
        lda_rep_mat[index, numb] = score

# Get topic distributions for each time slice
lda_topics_over_time = np.split(lda_rep_mat, num_time_slices, axis=1)

# Average topic distributions over time.
lda_avg_topics_over_time = [np.mean(topics, axis=1) for topics in lda_topics_over_time]

# Normalize to get the distribution over topics for each day.
lda_avg_topics_over_time = np.array(
    [topics / np.sum(topics) for topics in lda_avg_topics_over_time]
)

# Condense topic representations.
print("Printing html keywords (latex) table...")
topics_freqs = []
for i, topic in enumerate(lda_model.get_topics()):
    topics_freqs.append(
        {dictionary[i]: topic[i] for i in reversed(topic.argsort()[-20:])}
    )
topics_freqs = utils.condense_topic_keywords(topics_freqs)

# Make word and frequency lists for each topic.
sorted_topics = [
    sorted(topic.items(), key=lambda item: item[1], reverse=True)
    for topic in topics_freqs
]
word_lists = [[item[0] for item in topic[:num_keywords]] for topic in sorted_topics]
freq_lists = [[item[1] for item in topic[:num_keywords]] for topic in sorted_topics]
print(plotting.shaded_latex_topics(freq_lists, word_lists))

## Visualize topic distributions
start = datetime.datetime(2020, 2, 1, 0)
dates = [i * datetime.timedelta(days=1) + start for i in range(num_time_slices)]
date_strs = [date.strftime("%m-%d") for date in dates]
y_tick_labels = [
    "{}: {}".format(", ".join(word_lists[i][0:3]), i + 1) for i in range(rank)
]

fig, ax = plotting.heatmap(
    lda_avg_topics_over_time.T,
    x_tick_labels=date_strs,
    x_label="Date",
    y_tick_labels=y_tick_labels,
    y_label="Topic",
)

# Save figure.
fig_filename = "LDA_tweet_representation_of_topics_{}".format(data_name)
fig_filepath = os.path.join(local_path, fig_filename)
plotting.save_figure(fig, filepath=fig_filepath)
