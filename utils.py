#%%
import sys
import os
import re
# from stop_words import get_stop_words
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')

# Set of stopwords
stop_words = list(stopwords.words('english')) #About 150 stopwords
DEFAULT_TRAINING_PATH = "data/american/training/"

# Function to get, for each file inside the training directory, author, length and dictionary with words
def get_documents(feature_type="words", ngram_size=None, training_path = DEFAULT_TRAINING_PATH):
    documents = {}
    # Scanning all files inside the training path
    files = [os.path.join(training_path, f)
             for f in os.listdir(training_path)
             if os.path.isfile(os.path.join(training_path, f))]
    for f in files:
        # Processing the document as ngrams (groups of n letters)
        if feature_type == "chars":
            author, doc_length, words = process_ngrams(f, ngram_size)
        # Processing the document as words (separated by space)
        elif feature_type == "words":
            author, doc_length, words = process_words(f)
        documents[f] = [author, doc_length, words]
    return documents

# Extracts the words found in the documents, with duplicates
def extract_vocab(documents):
    vocabulary = []
    for values in documents.values():
        vocabulary += list(values[2].keys())
    return vocabulary

# Saves for each term the conditional probability it belongs to
# the preferred author and then returns the top n terms
# with the highest probability
def top_cond_probs_by_author(conditional_probabilities, author, n):
    cps = {}
    for term, probs in conditional_probabilities.items():
        cps[term] = probs[author]
    c = 0
    for term in sorted(cps, key=cps.get, reverse=True):
        if c < n:
            print(c, term, "score:", cps[term])
            c += 1
        else:
            break
    
#%%
# Get author and text of the book with the specified filename
def get_author_and_text(filename):
    f = open(filename, 'r', encoding='utf-8', errors="replace")
    c = 0
    sentences = ""
    for l in f.readlines():
        # Get the author's name
        if l.startswith("Author: "):
            author = l.split(":")[-1].rstrip().lstrip()
        
        # Starting the actual text of the book
        if l.startswith("*** START OF "):
            c=1
            continue 
        
        # Inserting sentences into the string or ending the cycle
        if c==1:
            if l.startswith("*** END OF"):
                break
            else:
                sentences = sentences+l
                
    # Lower case and removing symbols
    sentences = re.sub(r'[^a-z\s]', '', sentences.lower())
    sentences = re.sub(r'[\s]'," ", sentences)
    sentences = re.sub(' +', ' ', sentences)
    return author, sentences

def remove_stopwords(sentence):
    # Removing stopwords
    tokens_without_sw = [word for word in sentence.split()
                         if word not in stop_words]
    return tokens_without_sw

# Create a dictionary with words as key and their 
# counter of occurrences as value
def process_words(filename):
    # Get author and book text
    author, sentences = get_author_and_text(filename) 
    sentences = remove_stopwords(sentences)
    words = {}
    for w in sentences:
        if w in words:
            words[w] += 1
        else:
            words[w] = 1
    return author, len(words), words

# Create dictionary of ngrams, where n is inserted by the user, whose
# keys are ngrams and values are their counter
def process_ngrams(filename, n=3):
    # Get author and book text
    author, sentences = get_author_and_text(filename)   
    sentences = ' '.join(remove_stopwords(sentences))
    ngrams = {}
    for i in range(len(sentences)-n):
        ngram = sentences[i:i+n]
        if ngram in ngrams:
            ngrams[ngram] += 1
        else:
            ngrams[ngram] = 1
    return author, len(ngrams), ngrams
#%%
process_ngrams("data/american/test/melville_moby_dick.txt")
# %%
