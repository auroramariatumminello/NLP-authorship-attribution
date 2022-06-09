#%%
# Bag of words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

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

import nltk
from nltk.stem.snowball import SnowballStemmer
stemmer = nltk.stem.SnowballStemmer('english')
stop_words = set(nltk.corpus.stopwords.words('english'))
import re
def preprocessing(text):
   tokens = [word for word in nltk.word_tokenize(text)]
   tokens = map(str.lower, tokens)
   # Not sure about stemming
   # stems = [stemmer.stem(item) for item in tokens]
   return tokens

vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5,
#  ngram_range=(1, 3))
models = {
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    "LogisticRegression": LogisticRegression(random_state=0),
    "LinearSVC": LinearSVC(),
}
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
        df["text"], df["author"], stratify=df.author, test_size=0.3, random_state=30
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
def create_and_fit_model(vectors, classes, model="MultinomialNB"):
    try:
        classifier = models[model]
    except KeyError:
        # Classifier
        classifier = MultinomialNB()
    # Train the classifier:
    classifier.fit(vectors, classes)
    return classifier


#%%

# Read authors and texts
texts, authors = get_authors_and_texts(text_path)

# Split texts into train and test (randomly, considering at
# least one text per each author in the test set)
X_train, X_test, y_train, y_test = train_test_texts_split(texts, authors)

# Creation of the associated labels
classes, author_dict = create_labels_vector(y_train, authors_information_path)

# Bag of words representation
vectors = vectorizer.fit_transform(X_train)


# Model creation and fitting
classifier = create_and_fit_model(vectors, classes, model="MultinomialNB")
#%%

def predict_author_from_test_path(
    test_file, author_dict=author_dict, vectorizer=vectorizer
):
    with open(test_file, "r", errors="replace", encoding="utf-8") as f:
        test_text = f.read()
    mystery_vector = vectorizer.transform([test_text])
    # predict author's label
    predictions = classifier.predict(mystery_vector)
    return author_dict[int(predictions)]


def predict_author(test_text, author_dict=author_dict, vectorizer=vectorizer):
    mystery_vector = vectorizer.transform([test_text])
    # predict author's label
    predictions = classifier.predict(mystery_vector)
    return author_dict[int(predictions)]


#%%
def predict_authors(test_files, author_dict=author_dict):
    y_pred = []
    for test_file in test_files:
        y_pred.append(predict_author(test_file, author_dict))
    return y_pred


def compute_accuracy(y, pred):
    return round(sum(y == pred) / len(y), 2)

import pickle
def compare_predictions(y, pred):
    misclassified = pickle.load(open("misclassifications.pkl","rb"))
    assert len(y) == len(pred)
    for i in range(len(y)):
        if y[i] != pred[i]:
            misclassified.append([y[i],pred[i]])
    pickle.dump(misclassified,open("misclassifications.pkl","wb"))
    return pd.DataFrame({"y": y, "prediction": pred})


#%%
# Prediction
y_pred = predict_authors(X_test)

#%%
compute_accuracy(y_test, y_pred)
#%%
compare_predictions(y_test, y_pred)
#%%
import warnings
warnings.filterwarnings('ignore')
# Cross validation

# 5 Cross-validation
from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(5, shuffle=True, random_state=0)
models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    DecisionTreeClassifier(max_depth = 2)
]

classes, author_dict = create_labels_vector(authors, authors_information_path)
# Bag of words representation
vectors = vectorizer.fit_transform(texts)

entries = []
from tqdm import tqdm
for model in tqdm(models):
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, vectors, classes, cv=k_fold, n_jobs=1)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=["model_name", "fold_idx", "accuracy"])
#%%
cv_df.groupby(['model_name']).mean()
cv_df.groupby(['model_name']).median()

# Results suggest to use Multinomial NB or Logistic regression.
# What if we insert also authors features? What if we augment the number of
# ngrams to consider? or if we preprocess the text? We may lose the personalization
# of authors...