#%%
"""
AUTHORSHIP ATTRIBUTION

Usage:
  attribution.py [--test_file=<filename>] [--training_path=<filename>] [--tfidf] [--prob]
  attribution.py (-h | --help)

Options:
  -h --help                     Show this screen.
  --test_file <filename>        Specify path of the test file
  --training_path <filename>    Specify path of the training folder
  --tfidf                       Represent the text with tfidf
  --prob                        If activated, it shows the list of most likely authors
"""

from utils import *
from docopt import docopt

if __name__ == "__main__":
    arguments = docopt(__doc__, version="Authorship Attribution")


# ARGUMENTS
probabilities = True if arguments["--prob"] else False
vectorizer_type = "tfidf" if arguments["--tfidf"] else "bow"
test_file_path = arguments["--test_file"] if arguments["--test_file"] else None
TEXT_PATH = arguments["--training_path"] if arguments["--training_path"] else TEXT_PATH
AUTHORS_INFORMATION_PATH = TEXT_PATH + "/authors.csv"


# 1. Read authors and texts and exclude the test file from
# training set if it is inside the training path
print("Reading all texts...")
texts, authors = get_authors_and_texts(TEXT_PATH, test_file_path)

# 2. If there is no test file specified, call train test split
if test_file_path is None:
    print("Splitting texts into training and testing...")
    X_train, X_test, y_train, y_test = train_test_texts_split(texts, authors)

    authors = y_train
    texts = X_train
        

# 3. Creation of the associated labels
print("Creating authors labels...")
classes, author_dict = create_labels_vector(authors, AUTHORS_INFORMATION_PATH)

# 4. Bag of words representation
print("Representing the text as vectors...")
vectorizer, vectors = generate_text_vectors(texts, vectorizer_type)

# 5. Model creation and fitting
print("Model creation and fitting...")
classifier = create_and_fit_model(vectors, classes, model=MODEL_NAME)

if test_file_path is None:
    # 6. Prediction
    y_pred = predict_authors(X_test, author_dict, vectorizer, classifier)
    print("Accuracy: "+str(compute_accuracy(y_test, y_pred)))
    
    # Print probabilities
    if probabilities:
        print(get_results_df(X_test, y_test, author_dict, vectorizer, classifier)[['author_1','author_2','author_3','real_author','is_right','is_in_top3']])
else:
    # 6. Prediction
    actual_author, title, test_text, y_pred = predict_author_from_test_path(test_file_path, author_dict,
                                                                 vectorizer, classifier)
    
    if actual_author not in author_dict.values():
        print("\nAttention! The book you selected was written by none of the authors the model knows, so... let's see if it picks up some similar authors!\n")
    # Print book information
    print("\n================================\n")
    print("Title of the book:\t"+title)
    print("Predicted author:\t"+convert_label_to_complete_name([y_pred], AUTHORS_INFORMATION_PATH)[0])
    print("The real author:\t"+actual_author)
    
    # Print probabilities
    if probabilities:
        print("\n================================\n")
        _, message = predict_top_3_authors(test_text, author_dict,
                                        vectorizer, classifier)
        print(message)

    
