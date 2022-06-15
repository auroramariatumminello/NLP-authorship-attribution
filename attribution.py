#%%
"""
AUTHORSHIP ATTRIBUTION

Usage:
  attribution.py --file <filename> 
  attribution.py --chars=<n> <filename>
  attribution.py (-h | --help)
  attribution.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --words
  --chars=<kn>  Length of char ngram [default: 3].

"""
from utils import *
from termcolor import colored
from docopt import docopt

# if __name__ == '__main__':
#     arguments = docopt(__doc__, version='Authorship Attribution')

# if arguments["--words"]:
#     feature_type = "words"
# elif arguments["--chars"]:
#     feature_type = "chars"
#     ngram_size = int(arguments["--chars"])
# testfile = arguments["<filename>"]

#%%

# 1. Read authors and texts
print(colored("Reading all texts...",'blue'))
texts, authors = get_authors_and_texts(TEXT_PATH)

# 2. Split texts into train and test (randomly, considering at
# least one text per each author in the test set)
print(colored("Splitting texts into training and testing...",'blue'))
X_train, X_test, y_train, y_test = train_test_texts_split(texts, authors)

# 3. Creation of the associated labels
print(colored("Creating authors labels...",'blue'))
classes, author_dict = create_labels_vector(y_train, AUTHORS_INFORMATION_PATH)


# 4. Bag of words representation
print(colored("Representing the text as vectors...",'blue'))
vectorizer, vectors = generate_text_vectors(X_train,'bow')


# 5. Model creation and fitting
print(colored("Model creation and fitting...",'blue'))
classifier = create_and_fit_model(vectors, classes, model=MODEL_NAME)

#%%
# 6. Prediction
print(colored("Prediction!",'blue'))
y_pred = predict_authors(X_test, author_dict, vectorizer, classifier, prob=True)

##########################################
# Additional step: 
# Compare within the top 3 more likely authors
##########################################
prediction_heatmap(X_test, y_test,author_dict, vectorizer, classifier)
#%%
