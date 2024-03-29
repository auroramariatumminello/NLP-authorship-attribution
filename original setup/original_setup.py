#%%
"""
AUTHORSHIP ATTRIBUTION

Usage:
  attribution.py --words <filename>
  attribution.py --chars=<n> <filename>
  attribution.py (-h | --help)
  attribution.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --words
  --chars=<kn>  Length of char ngram [default: 3].

"""

#%%
# Libraries
import sys
import os
import math
from utils import get_documents, extract_vocab, process_ngrams, process_words, top_cond_probs_by_author
from docopt import docopt

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Authorship Attribution 2.1')
#%%
# Default values for hyperparameters
feature_type = "words"
ngram_size = 5
testfile = "../data/american/test_books/hawthorne_dead.txt"

if arguments["--words"]:
    feature_type = "words"
elif arguments["--chars"]:
    feature_type = "chars"
    ngram_size = int(arguments["--chars"])
testfile = arguments["<filename>"]

#%%

alpha = 0.0001
import pandas as pd
authors = pd.read_csv("../data/american/authors.csv")
classes = authors['label']
#%%

#%%
# Counts the number of documents found
def count_docs(documents):
    return len(documents)

# Count how many documents belong to a class
def count_docs_in_class(documents, c):
    count=0
    for values in documents.values():
        if values[0] == c:
            count+=1
    return count

# Contatenate all vocabularies belonging to the same class
def concatenate_text_of_all_docs_in_class(documents,c):
    words_in_class = {}
    for d,values in documents.items():
        if values[0] == c:
            words_in_class.update(values[2])
    return words_in_class

#%%
# Naive Bayes Classifier
def train_naive_bayes(classes, documents):
    # Get all words in all documents
    vocabulary = extract_vocab(documents)
    conditional_probabilities = {}
    
    # For each word, the dictionary is initialized
    for t in vocabulary:
        conditional_probabilities[t] = {}
    priors = {}
    print("\n***\nCalculating priors and conditional probabilities for each class...\n***")
    for c in classes:
        # Prior probability (number of documents of that class over the total number of docs)
        class_size = count_docs_in_class(documents, c)
        priors[c] = class_size / count_docs(documents)
        
        # Set of words inside all documents of the same class
        words_in_class = concatenate_text_of_all_docs_in_class(documents,c)

        denominator = sum(words_in_class.values())
        for t in vocabulary:
            if t in words_in_class:
                conditional_probabilities[t][c] = (words_in_class[t] + alpha) / (denominator * (1 + alpha))
            else:
                conditional_probabilities[t][c] = (0 + alpha) / (denominator * (1 + alpha))
    return vocabulary, priors, conditional_probabilities


def apply_naive_bayes(classes, priors, conditional_probabilities, test_document):
    scores = {}
    if feature_type == "chars":
        author, _, words = process_ngrams(test_document,ngram_size)
    elif feature_type == "words":
        author, _, words = process_words(test_document)
        
    print("Top words inside ",test_document,": \n",sorted(words.items(), key=lambda x:-x[1])[:10])
    for c in classes:
        scores[c] = math.log(priors[c])
        for t in words:
            if t in conditional_probabilities:
                for i in range(words[t]):
                    scores[c] += math.log(conditional_probabilities[t][c])
    print("\nScores in descending order:")
    for author in sorted(scores, key=scores.get, reverse=True):
        print(author,"\t:",scores[author])
        
        
# Naive Bayes Training
import pickle

# Reading documents
print("***\nReading documents...\n***")
documents = get_documents(feature_type, ngram_size)
# Computing training data
print("***\nComputing probabilities...\n***\n")
vocabulary, priors, conditional_probabilities = train_naive_bayes(classes, documents)

# Saving training results
voc = 'vocabulary.txt'
pri = 'priors.txt'
cond = 'conditional_probabilities.txt'
pickle.dump(vocabulary, open(voc, 'wb'))
pickle.dump(priors, open(pri, 'wb'))
pickle.dump(conditional_probabilities, open(cond, 'wb'))

# print("***\nBest features for each author...\n***\n")
# for author in classes:
#     print("\nBest features for",author)
#     top_cond_probs_by_author(conditional_probabilities, author, 5)
#     print("\n")

print("***\nNaive Bayes Classifier...\n***\n")
apply_naive_bayes(classes, priors, conditional_probabilities, testfile)
