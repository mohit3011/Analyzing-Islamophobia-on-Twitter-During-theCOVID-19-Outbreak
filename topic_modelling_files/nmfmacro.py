import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
import nltk
from pprint import pprint
import pandas as pd
import json
from collections import Counter, OrderedDict
from dateutil import tz
from datetime import datetime
import numpy as np
import os
import seaborn as sns
import glob, os
sns.set()
import gensim
from tqdm import tqdm
import re
import gzip
import sklearn
import bokeh
from gensim import corpora
import pickle
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer,PorterStemmer
import matplotlib.pyplot as plt
import re, string, unicodedata
import pyLDAvis.gensim
from gensim.models import CoherenceModel
from top2vec import Top2Vec
import spacy
nlp = spacy.load('en_core_web_sm')
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~[(){}<>""\',=`;:\[\]\?\\/|_]'''
stop_words = set(stopwords.words('english'))
directorylist=['../2020-02','../2020-03','../2020-04','../2020-05']
numbers = re.compile(r'(\d+)')

def top_words(topic, n_top_words):
    return topic.argsort()[:-n_top_words - 1:-1]


def topic_table(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        t = (topic_idx)
        topics[t] = [feature_names[i] for i in top_words(topic, n_top_words)]
    return pd.DataFrame(topics)


def whitespace_tokenizer(text):
    pattern = r"(?u)\b\w\w+\b"
    tokenizer_regex = RegexpTokenizer(pattern)
    tokens = tokenizer_regex.tokenize(text)
    return tokens


# Funtion to remove duplicate words
def unique_words(text):
    ulist = []
    [ulist.append(x) for x in text if x not in ulist]
    return ulist


def word_count(text):
    return len(str(text).split(' '))

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

'''Preprocessing functions'''

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    new_words = []
    for word in words:
        n_word = re.sub(r'[(){}<>""\',=`;:\[\]\?\\/|_]!-@#$%^&*~', ' ', word)
        #new_word = re.sub(r'[_]',' ',n_word)
        if n_word != '':
            new_words.append(n_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words
#
def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = PorterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

'''Call either lemmatization or stemming'''
# def lemmatize_verbs(words):
#     """Lemmatize verbs in list of tokenized words"""
#     lemmatizer = WordNetLemmatizer()
#     lemmas = []
#     for word in words:
#         lemma = lemmatizer.lemmatize(word, pos='v')
#         lemmas.append(lemma)
#     return lemmas
#
'''Parent functions'''
def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    # stems = stem_words(words)
    words = [each for each in words if len(each) > 1]
    words = [each for each in words if ' ' not in each]
    return words

def stemming(words):
    stems = stem_words(words)
    return stems

# def lemmatize(words):
#     lemmas = lemmatize_verbs(words)
#     return lemmas
#
# ''' End of Functions '''

''' Extracting Tweets '''
tweets = []
for i in directorylist:
    for infile in sorted(glob.glob(i+'/*.jsonl.gz'), key=numericalSort):
        json_content = []
        with gzip.open(infile , 'rb') as gzip_file:
            for line in gzip_file:  # Read one line.
                line = line.rstrip()
                if line:  # Any JSON data on it?
                    obj = json.loads(line)
                    json_content.append(obj)
            for tweet in json_content:
                if 'full_text' in tweet.keys():
                    text=""
                    if 'retweeted_status' in tweet.keys():
                        text = tweet['retweeted_status']['full_text']
                    else:
                        text = tweet['full_text']
                    no_punct=""
                    for char in text:
                        if char not in punctuations:
                            no_punct = no_punct + char
                    word_tokens = word_tokenize(no_punct)
                    words=normalize(word_tokens)
                    tweets.append(words)

''' Common Words '''
p_text = [item for sublist in tweets for item in sublist]
top_20 = pd.DataFrame(
    Counter(p_text).most_common(20),
    columns=['word', 'frequency']
)
print(top_20)

fig = plt.figure(figsize=(20,7))

g = sns.barplot(
    x='word',
    y='frequency',
    data=top_20,
    palette='GnBu_d'
)

g.set_xticklabels(
    g.get_xticklabels(),
    rotation=45,
    fontsize=14
)

plt.yticks(fontsize=14)
plt.xlabel('Words', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Top 20 Words', fontsize=17)

file_name = 'top_words'

fig.savefig(
    file_name + '.png',
    dpi=fig.dpi,
    bbox_inches='tight'
)

# plt.show()

# Create a dictionary
# In gensim a dictionary is a mapping between words and their integer id
dictionary = Dictionary(tweets)

# Filter out extremes to limit the number of features
dictionary.filter_extremes(
    no_below=3,
    no_above=0.85,
    keep_n=5000
)

# Create the bag-of-words format (list of (token_id, token_count))
corpus = [dictionary.doc2bow(text) for text in tweets]

# Create a list of the topic numbers we want to try
topic_nums = list(np.arange(5, 75 + 1, 5))

# Run the nmf model and calculate the coherence score
# for each number of topics
coherence_scores = []

for num in topic_nums:
    nmf = Nmf(
        corpus=corpus,
        num_topics=num,
        id2word=dictionary,
        chunksize=2000,
        passes=5,
        kappa=.1,
        minimum_probability=0.01,
        w_max_iter=300,
        w_stop_condition=0.0001,
        h_max_iter=100,
        h_stop_condition=0.001,
        eval_every=10,
        normalize=True,
        random_state=42
    )

    # Run the coherence model to get the score
    cm = CoherenceModel(
        model=nmf,
        texts=tweets,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence_scores.append(round(cm.get_coherence(), 5))

# Get the number of topics with the highest coherence score
scores = list(zip(topic_nums, coherence_scores))
best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]
print(best_num_topics)
# Plot the results
fig = plt.figure(figsize=(15, 7))

plt.plot(
    topic_nums,
    coherence_scores,
    linewidth=3,
    color='#4287f5'
)

plt.xlabel("Topic Num", fontsize=14)
plt.ylabel("Coherence Score", fontsize=14)
plt.title('Coherence Score by Topic Number - Best Number of Topics: {}'.format(best_num_topics), fontsize=18)
plt.xticks(np.arange(5, max(topic_nums) + 1, 5), fontsize=12)
plt.yticks(fontsize=12)

file_name = 'c_score'

fig.savefig(
    file_name + '.png',
    dpi=fig.dpi,
    bbox_inches='tight'
)


# Create the tfidf weights
tfidf_vectorizer = TfidfVectorizer(
    min_df=3,
    max_df=0.85,
    max_features=5000,
    ngram_range=(1, 2),
    preprocessor=' '.join
)

tfidf = tfidf_vectorizer.fit_transform(tweets)

# Save the feature names for later to create topic summaries
tfidf_fn = tfidf_vectorizer.get_feature_names()

# Run the nmf model
nmf = NMF(
    n_components=best_num_topics,
    init='nndsvd',
    max_iter=500,
    l1_ratio=0.0,
    solver='cd',
    alpha=0.0,
    tol=1e-4,
    random_state=42
).fit(tfidf)
# plt.show()

docweights = nmf.transform(tfidf_vectorizer.transform(tweets))

n_top_words = 15

topic_df = topic_table(
    nmf,
    tfidf_fn,
    n_top_words
).T

# Cleaning up the top words to create topic summaries
topic_df['topics'] = topic_df.apply(lambda x: [' '.join(x)], axis=1) # Joining each word into a list
topic_df['topics'] = topic_df['topics'].str[0]  # Removing the list brackets
topic_df['topics'] = topic_df['topics'].apply(lambda x: whitespace_tokenizer(x)) # tokenize
topic_df['topics'] = topic_df['topics'].apply(lambda x: unique_words(x))  # Removing duplicate words
topic_df['topics'] = topic_df['topics'].apply(lambda x: [' '.join(x)])  # Joining each word into a list
topic_df['topics'] = topic_df['topics'].str[0]  # Removing the list brackets

topic_df.to_csv('topics.csv')

topic_df = topic_df['topics'].reset_index()
topic_df.columns = ['topic_num', 'topics']


df_temp = pd.DataFrame({
    'tweets': tweets,
    'topic_num': docweights.argmax(axis=1)
})

df_topics = df_temp.merge(
    topic_df,
    on='topic_num',
    how='left'
)


A = tfidf_vectorizer.transform(tweets)
W = nmf.components_
H = nmf.transform(A)

print('A = {} x {}'.format(A.shape[0], A.shape[1]))
print('W = {} x {}'.format(W.shape[0], W.shape[1]))
print('H = {} x {}'.format(H.shape[0], H.shape[1]))

r = np.zeros(A.shape[0])

for row in range(A.shape[0]):
    r[row] = np.linalg.norm(A[row, :] - H[row, :].dot(W), 'fro')

sum_sqrt_res = round(sum(np.sqrt(r)), 3)
print("Sum of the squared residuals is : ")
print(sum_sqrt_res)
print(len(r))


df_topics['resid'] = r
df_topics.to_csv('topicstweets.csv')
resid_data = df_topics[[
    'topic_num',
    'resid'
]].groupby('topic_num').mean().sort_values(by='resid')

# Plot a bar chart for the avg. residuls by topic
fig = plt.figure(figsize=(20,7))

x = resid_data.index
y = resid_data['resid']

g = sns.barplot(
    x=x,
    y=y,
    order=x,
    palette='rocket'
)

g.set_xticklabels(
    g.get_xticklabels(),
    fontsize=14
)

plt.yticks(fontsize=14)
plt.xlabel('Topic Number', fontsize=14)
plt.ylabel('Residual', fontsize=14)
plt.title('Avg. Residuals by Topic Number', fontsize=17)

file_name = 'avg_resid'

fig.savefig(
    file_name + '.png',
    dpi=fig.dpi,
    bbox_inches='tight'
)
