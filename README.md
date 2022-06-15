# 📚 Authorship attribution


This repository shows how it is possible to discover the author of a text book, given a list of authors among which we can choose. It may be resused for plagiarism detection or to discover the author of messages, by changing the list of authors and the directory with txt documents necessary for the training phase. 

The model will try to guess the true author of the text you choose (or some random ones the model has never seen taken from the directory of predownloaded books). 

## Requirements

If you want to execute the code, it is suggested to create a conda environment where to install the [requirements](requirements.txt): 

If you're new to NLTK, you should also run: 

    nltk.download("punkt")
    nltk.download("stopwords")

## Run the code

To run the [main code](attribution.py), type:

    python attribution.py 

You can also switch to the Russian literature (or your personal folder):

    python attribution.py --training_path data/russian/ --test_file data/russian/test_books/dostoevsky_white_nights.txt

By adding --prob flag at the end, you can obtain:

* a dataframe with the probabilities per each author to have written a book, if you do not specify a test file;
* if you specify a test file, you will get the ordered list of authors, based on how likely they have written the book you requested. 

Whenever you do not specify a test file, you will only get the accuracy (unless you add --prob). Otherwise, you get the book information (title, author, predicted author). 

Also, you can switch from bag of words representation to tfidf by specifying --tfidf:

    python attribution.py --tfidf

You can also test the [Jupyter Notebook](attribution.ipynb), already executed, and change the paths to get your predictions. 


## Results

Multiple models are suggested in order to classify a text, but mainly MultinomialNB is used. Why?

### Leave One Out Cross Validation

If we run a [cross validation](cross_validation.py) with the leave one out k-fold, this means that the model will train on $n-1$ texts, leaving $1$ to testing. The mean accuracies are showed in the table below, shoing that MultinomialNB is the best model, followed by LinearSVC and LogisticRegression.

Note that in case of the DecisionTreeClassifier performance are comparable with a random model that assigns a book to a random author (1/number of authors).


| **model_name**         | **mean accuracy** |
|------------------------|--------------|
| MultinomialNB          | 0.911111     |
| LinearSVC              | 0.877778     |
| LogisticRegression     | 0.844444     |
| RandomForestClassifier | 0.633333     |
| DecisionTreeClassifier | 0.077778     |

### K fold Cross Validation

The cross validation script also presents a proper K-fold cross validation with 5 splits, which means that it tries 5 different configurations of training/testing sets. You can find the results below, considered as mean and median accuracy. If we consider the mean accuracy, LogisticRegression performs slightly better than MultinomialNB, which instead is the best model when looking at the median accuracy. 


| **model_name**         | **mean accuracy** |
|------------------------|--------------|
| LogisticRegression     | 0.844444     |
| MultinomialNB          | 0.811111     |
| LinearSVC              | 0.800000     |
| RandomForestClassifier | 0.577778     |
| DecisionTreeClassifier | 0.122222     |


| **model_name**         | **median accuracy** |
|------------------------|--------------|
| MultinomialNB          | 0.833333     |
| LinearSVC              | 0.777778     |
| LogisticRegression     | 0.777778     |
| RandomForestClassifier | 0.555556     |
| DecisionTreeClassifier | 0.111111     |