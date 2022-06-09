#%%
# Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# Train test split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# Utility
import os
import pandas as pd
from sympy import comp

#%%
text_path = "data/american"
authors_information_path = "data/authors.csv"
#%%


def load_texts(training_path=text_path):
    # Load all training files' paths
    files = [
        os.path.join(training_path, f)
        for f in os.listdir(training_path)
        if os.path.isfile(os.path.join(training_path, f))
    ]
    return files


def train_test_texts_split(texts, authors):
    df = pd.DataFrame({"text": texts, "author": authors})
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["author"], stratify=df.author, test_size=0.3
    )
    return X_train, X_test, y_train, y_test


def get_authors_and_texts(training_path=text_path):
    files = load_texts(training_path)
    # Save the author of each document and the document's text
    texts = []
    authors = []
    for name in files:
        authors.append(name.split("\\")[-1].split("_")[0])
        f = open(name, "r", encoding="utf-8", errors="replace")
        texts.append(f.read())
    return texts, authors


#%%


def create_labels_vector(authors, authors_information_path=authors_information_path):
    authors_df = pd.read_csv(authors_information_path)
    authors_df.label = pd.Categorical(authors_df.label)
    num_auth = dict(enumerate(authors_df["label"].cat.categories))
    # print(num_auth)
    auth_num = {v: k for k, v in num_auth.items()}
    # print(auth_num)
    #%%
    classes = [auth_num[author] for author in authors]
    return classes, num_auth

#%%
# Read authors and texts
texts, authors = get_authors_and_texts(text_path)

#%%
# PREPROCESSING
import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = set(nltk.corpus.stopwords.words('english'))
import re

def preprocessing(text):
   tokens = [word for word in nltk.word_tokenize(text) if (len(word) > 3 and len(word.strip('Xx/')) > 2 and len(re.sub('\d+', '', word.strip('Xx/'))) > 3) ] 
   tokens = map(str.lower, tokens)
   stems = [stemmer.stem(item) for item in tokens if (item not in stop_words)]
   return stems

#%%

vectorizer_tf = TfidfVectorizer(tokenizer=preprocessing, stop_words=None, 
                                max_df=0.75, max_features=1000, 
                                lowercase=False, ngram_range=(1,2))

# Split texts into train and test (randomly, considering at
# least one text per each author in the test set)
X_train, X_test, y_train, y_test = train_test_texts_split(texts, authors)

# Creation of the associated labels
classes, author_dict = create_labels_vector(y_train, authors_information_path)

# Bag of words representation
train_vectors = vectorizer_tf.fit_transform(X_train)
test_vectors = vectorizer_tf.fit_transform(X_test)

# Converting sets into array
train_df=pd.DataFrame(train_vectors.toarray(), columns=vectorizer_tf.get_feature_names())
test_df=pd.DataFrame(test_vectors.toarray(), columns=vectorizer_tf.get_feature_names())

train_df['AUTHOR_']= list(y_train)
test_df['AUTHOR_'] = list(y_test)

#%%
import h2o
from h2o.automl import H2OAutoML

# Initializing the cluster...
h2o.init()
#%%
# Creating H2O dataframes for test and training
h2o_train_df = h2o.H2OFrame(train_df)
h2o_test_df = h2o.H2OFrame(test_df)

# Converting the author to factor column
h2o_train_df['AUTHOR_'] = h2o_train_df['AUTHOR_'].asfactor()
h2o_test_df['AUTHOR_'] = h2o_test_df['AUTHOR_'].asfactor()
#%%
aml = H2OAutoML(max_models = 5, seed = 10, 
                exclude_algos = ["StackedEnsemble"], verbosity="info", nfolds=2, balance_classes=False, max_after_balance_size=0.3)

x=vectorizer_tf.get_feature_names()
y='AUTHOR_'

#%%
aml.train(x = x, y = y, training_frame = h2o_train_df, validation_frame=h2o_test_df)
#%%
aml.leaderboard

#%%
pred=aml.leader.predict(h2o_test_df)
#%%
import numpy as np
import matplotlib.pyplot as plt
pred = pred.as_data_frame()
pred.sort_values('predict', inplace=True)
pred = pred.set_index('predict')

# Have a look at authors considered as similar
import seaborn as sns
sns.heatmap(pred, cmap='Blues')