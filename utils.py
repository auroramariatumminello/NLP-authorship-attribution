#%%
# Bag of words and tfidf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

# Train test split
from sklearn.model_selection import train_test_split
from random import randint

# Utility
import os
import pandas as pd
import re
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


# VARIABLES
import nltk
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
STOP_WORDS: list = set(nltk.corpus.stopwords.words("english"))
vectorizer_type: str = 'bow'

# List of models within which we can choose for multiclassification
models: dict = {
    "MultinomialNB": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5),
    "DecisionTree": DecisionTreeClassifier(max_depth=2),
    "LogisticRegression": LogisticRegression(random_state=0),
    "LinearSVC": CalibratedClassifierCV(LinearSVC()),
}

# CONSTANTS
MODEL_NAME: str = "MultinomialNB"
TEXT_PATH: str = "data/american"
AUTHORS_INFORMATION_PATH: str = "data/american/authors.csv"
RANDOM_STATE = randint(0, 1000)

##########################################
# PHASE 1: Text and authors reading
##########################################
def load_texts(to_exclude, training_path: str = TEXT_PATH):
    """
    Given a training path, whose default is the TEXT_PATH variable,
    it searches for all txt files inside the specified path and returns
    a list with the complete path of all books.

        Parameters:
            training_path (str): path from which the function will get txt files' paths
            to_exclude (str): path of a file to exclude, because chosen as test file

        Returns:
            files (list): list of file paths inside the specified folder
    """
    # Load all training files' paths
    files = [
        os.path.join(training_path, f)
        for f in os.listdir(training_path)
        if os.path.isfile(os.path.join(training_path, f))
    ]
    to_exclude = "_-_-_" if to_exclude is None else to_exclude
    files = [f for f in files if f.endswith(".txt") and to_exclude not in f]
    return files


def get_authors_and_texts(training_path: str = TEXT_PATH, to_exclude = "_-_-_"):
    """
    Given the training path, whose default value is TEXT_PATH,
    the function loads texts paths and for each file path it
    extracts the author's label from the filename (first string)
    and the content of the file itself.

        Parameters:
            training_path (str): folder from which the function reads the texts
            to_exclude (str): path of a file to exclude, because chosen as test file


        Returns:
            texts (list): corpus of all the books found in the training path
            authors (list): associated authors of the texts
    """
    files = load_texts(to_exclude, training_path)
    # Save the author of each document and the document's text
    texts = []
    authors = []
    for name in files:
        authors.append(name.split("\\")[-1].split("_")[0])
        f = open(name, "r", encoding="utf-8", errors="replace")
        texts.append(f.read())
    texts = [remove_heading(text)[-1] for text in texts]
    return texts, authors


def remove_heading(text: str):
    """
    Given a text, the function removes the heading and the ending part
    automatically inserted in every Gutenberg book, delimited by
    *** START OF...*** and *** END ... ***

        Parameters:
            text (str): book text

        Returns:
            text (str): the book text without heading and ending part,
            with metadata and copyright information (mere book corpus).
            author, title (str:Optional): metadata extracted from the heading
    """
    try:
        heading, text = re.split("\*{3}\s?START OF .*\*{3}", text)
        title, heading = heading.split("Title: ")[-1].split("Author: ")
        author = heading.split("\n")[0]
        text = re.split("\*{3}\s?END.*\*{3}", text)[0]
        return title.strip(), author.strip(), text

    except:
        print("No heading apparently..")
        return text


##########################################
# PHASE 2: Split dataset of books into train and test sets
##########################################
def train_test_texts_split(texts: list, authors: list):
    """
    Given a list of texts and the list of the associated authors,
    the function splits the text into training and testing sets,
    saving the related authors inside y_train and y_test.
    To avoid an author being only in the training/testing set,
    the stratify parameter is set on the author itself.

        Parameters:
            texts (list): list of text books
            authors (list): list of the authors associated to the texts

        Returns:
            X_train (list): training set with texts
            y_train (list): list of the authors of the books in the training set
            X_test (list): testing set with texts
            y_test (list): list of the authors of the books in the testing set
    """
    test_size = 0.3
    if test_size*len(texts)<len(set(authors)):
        test_size = len(set(authors))/len(texts)
        
    df = pd.DataFrame({"text": texts, "author": authors})
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["author"],
        stratify=df.author,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


##########################################
# PHASE 3: Association of the author label to the text
##########################################
def create_labels_vector(authors, authors_information_path=AUTHORS_INFORMATION_PATH):
    '''
    Given the list of authors' labels and the path to the csv
    with the authors' information, the function will codify the labels
    as sequential numbers, such that for each author, there's a number (auth_num)
    and for each number there's an author's label (num_auth).
    
        Parameters: 
            authors (list): list of all the authors associated to texts
            authors_information_path (str): path to the csv with the relative information about authors
            
        Returns:
            classes (list): mapped list from authors to their match in numbers
            num_auth (dict): dictionary that maps a number to its author 
                            (useful during predictions to convert the number to string)
    '''
    authors_df = pd.read_csv(authors_information_path)
    authors_df.label = pd.Categorical(authors_df.label)
    num_auth = dict(enumerate(authors_df["label"].cat.categories))
    auth_num = {v: k for k, v in num_auth.items()}
    #%%
    classes = [auth_num[author] for author in authors]
    return classes, num_auth


##########################################
# PHASE 4: Text representation
# Choose your fighter! Bag of words or tfidf
##########################################
def preprocessing(text):
    '''
    Function to pass as tokenizer inside the vectorizer
    
        Parameters: 
            text (str)
            
        Returns:
            stems (list): list of words reduced to their root
    '''
    tokens = [word for word in nltk.word_tokenize(text)]
    stems = [stemmer.stem(item) for item in tokens]
    return stems

def generate_text_vectors(texts,vectorizer='bow', ngrams_max=4):
    '''
    Given a list of texts, the function creates a vectorizer, 
    based on the user preferences, and returns the vectorizer itself,
    plus the vectorial representation of the texts
    
        Parameters:
            texts (list): list of text corpus 
            vectorizer (str): if "bow", the text is represented through Bag of Words,
                                tf-idf otherwise
        
        Returns:
            vectorizer (CountVectorizer or TfidfVectorizer)
            vectors (sparse matrix): (sparse) matricial representation of the texts
    
    '''
    if vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(min_df=5,
                                     ngram_range=(1,ngrams_max), 
                                     tokenizer=preprocessing,
                                     stop_words=STOP_WORDS)
    else:
        vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectorizer, vectors

#%%

##########################################
# PHASE 5: Creation of the model
##########################################
def create_and_fit_model(vectors, classes, model="MultinomialNB"):
    '''
    Given the vectorial representation of texts and their classes, the 
    model is created, fitted and then returned. Whenever the specified model
    is not present in the dictionary, it raises a key error and retries with 
    the Multinomial Naive Bayes Classifier
    
        Parameters: 
            vectors (sparse matrix): vectorial representation of texts
            classes (list): numerical label of the associated author
            model (str): name of the model, to choose among the models'
                        dictionary inside this document
                        
        Returns:
            classifier (model): returns a fitted classifier 
    '''
    try:
        classifier = models[model]
    except KeyError:
        # Classifier
        classifier = MultinomialNB()
    # Train the classifier:
    classifier.fit(vectors, classes)
    return classifier


##########################################
# PHASE 6: Predictions
##########################################
def predict_author_from_test_path(test_file, author_dict, vectorizer, classifier, probabilities=False):
    '''
    Given a test file path, the function reads it and transforms it into its
    vectorial form through the vectorizer. Then it predicts its author through
    the help of the classifier. In the end, since the prediction is a number,
    the output of the model is mapped into the author's name. 
    
        Parameters:
            test_file (str): file path
            author_dict (dict): dictionary to map numbers to authors' labels
            vectorizer (CountVectorizer or TfidfVectorizer)
            classifier (model): model chosen among the list of models 
            probabilities (bool): chooses if returning the single author or the list of
                                    probabilities per author
        
        Returns: 
            author (str): Real Author of the chosen text 
            title (str): Title of the book extracted from the text file
            test_text (str): Text book
            author_pred (str): Predicted Author of the chosen text 
            
    '''
    with open(test_file, "r", errors="replace", encoding="utf-8") as f:
        test_text = f.read()
    title, author, test_text = remove_heading(test_text)
    mystery_vector = vectorizer.transform([test_text])
    
    if probabilities:
        try:
            predictions = classifier.predict_proba(mystery_vector).tolist()
            predictions = [y for x in predictions for y in x]
            return author, title, predictions
        except:
            print("The model has no predict_proba method")    
    predictions = classifier.predict(mystery_vector)
    return author, title, test_text, author_dict[int(predictions)]


def predict_author(test_text, author_dict, vectorizer, classifier, probabilities=False):
    '''
    Given a text, the function transforms it into its vectorial form
    through the vectorizer. Then two different cases are presented:
    
    - The model predicts the text author through and since the prediction 
        is a number, the output of the model is mapped into the author's name. 
    - If probabilities flag is on, the model predicts, for each author,
        the probability that he/she has written the selected text.
    
        Parameters:
            test_text (str): text
            author_dict (dict): dictionary to map numbers to authors' labels
            vectorizer (CountVectorizer or TfidfVectorizer)
            classifier (model): model chosen among the list of models 
        
        Returns: 
            author (str): Author of the chosen text 
            or
            predictions (list): list of probabilities, ordered by author's label
    '''
    mystery_vector = vectorizer.transform([test_text])
    if probabilities:
        try:
            predictions = classifier.predict_proba(mystery_vector).tolist()
            predictions = [y for x in predictions for y in x]
            return predictions
        except:
            print("The model has no predict_proba method")    
    predictions = classifier.predict(mystery_vector)
    return author_dict[int(predictions)]


def predict_authors(test_files, author_dict, vectorizer, classifier, prob=False):
    '''
    Function to predict the author/probabilities per author of a list of texts.
    It returns a list of authors or a list of probabilities, one item per each text
    
        Parameters:
            test_files (list): multiple texts
            author_dict (dict): dictionary to map numbers to authors' labels
            vectorizer (CountVectorizer or TfidfVectorizer)
            classifier (model): model chosen among the list of models 
            prob (bool): if false, returns a list of authors, a list of probabilities
                        per author otherwise
                        
        Returns: 
            y_pred (list): list of authors associated to the test files in input
    '''
    y_pred = []
    for test_file in test_files:
        y_pred.append(predict_author(test_file, author_dict, vectorizer, classifier, probabilities=prob))
    return y_pred
    
    
##########################################
# Prediction utilities
##########################################
def predict_top_3_authors(test_file, author_dict, vectorizer, classifier, is_path=False):
    '''
    Given a test file (path or text), the function predicts the 
    probabilities per author and it returns both the predictions (sorted
    by most likely) and a message to print in the command prompt
    
        Parameters:
            test_file (str): file path or text
            author_dict (dict): dictionary to map numbers to authors' labels
            vectorizer (CountVectorizer or TfidfVectorizer)
            classifier (model): model chosen among the list of models 
            is_path (bool): whether the first parameter is a path or a text

        Returns:
            preds (list of lists): author and associated probability
            message (str): to print in the prompt
            
    '''
    from tabulate import tabulate
    if is_path:
        preds = predict_author_from_test_path(test_file, author_dict, vectorizer, classifier, probabilities=True)[-1]
    else:
        preds = predict_author(test_file, author_dict, vectorizer, classifier, probabilities=True)
    preds = sorted(zip(preds, author_dict.values()), reverse=True)
    results = [[x[1],x[0]] for x in preds if x[0]>0]
    message = "Authors that most likely have written the requested book:\n"
    message += tabulate(results, headers=['Author','Probability'])
    return preds, message



def get_results_df(X_test, y_test, author_dict, vectorizer, classifier):
    '''
    
    '''
    preds = predict_authors(X_test, author_dict, vectorizer, classifier, prob=True)
    author_preds = [
        sorted(zip(x, author_dict.values()), reverse=True)[:3] for x in preds
    ]

    author_preds = [x[-1] for sub in author_preds for x in sub]
    author_preds = np.reshape(author_preds, (len(X_test), 3))
    df = pd.DataFrame(preds, columns=author_dict.values())
    df[["author_1", "author_2", "author_3"]] = author_preds
    df["real_author"] = list(y_test)
    # If the author is the first one
    df["is_right"] = df["real_author"] == df["author_1"]

    # If the author is in the top 3
    df["is_in_top3"] = (
        (df["real_author"] == df["author_1"])
        | (df["real_author"] == df["author_2"])
        | (df["real_author"] == df["author_3"])
    )
    return df.sort_values(['real_author'])


def compute_accuracy(y, pred):
    return round(sum(y == pred) / len(y), 2)

def prediction_heatmap(X_test, y_test, author_dict, vectorizer, classifier):
    df = get_results_df(X_test, y_test, author_dict, vectorizer, classifier)
    print("Accuracy top 1: "+str(round(compute_accuracy(df["real_author"], df["author_1"]),2)))
    print("Accuracy top 3: "+str(round(sum(df["is_in_top3"] == True)/len(df),2)))

    # Heatmap to compare authors similarity
    df_heatmap = df.set_index("real_author").drop(
        ["author_1", "author_2", "author_3", "is_right", "is_in_top3"], axis=1
    )
    df_heatmap = df_heatmap.sort_index()

    plt.figure(figsize=(15, 10))
    cmap = sns.cm.rocket_r
    sns.heatmap(df_heatmap, cmap="magma_r", linewidths=0.5)
    plt.show()
    
    return df

def convert_label_to_complete_name(label: list, authors_info = AUTHORS_INFORMATION_PATH):
    '''
    Given a list of labels and the path to the csv with authors' information,
    it converts the label into the complete name of the authors
    
        Parameters:
            label (list)
            authors_info (str): path of csv with authors info
            
        Returns:
            list of complete names of labels
    '''
    authors = pd.read_csv(authors_info)
    return [authors.loc[authors['label']==l].iloc[0,0] for l in label]
    