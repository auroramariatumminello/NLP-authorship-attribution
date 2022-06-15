# 📚 Authorship attribution


This repository shows how it is possible to discover the author of a text book, given a list of authors among which we can choose. It may be resused for plagiarism detection or to discover the author of messages, by changing the list of authors and the directory with txt documents necessary for the training phase. 

The model will try to guess the true author of the text you choose (or some random ones the model has never seen taken from the directory of predownloaded books). 

## Requirements

It is suggested to create a 

## Run the code

To run the code, type:

    python attribution.py 

You can also switch to the Russian literature (or your personal folder):

    python attribution.py --training_path data/russian/ --test_file data/russian/test_books/dostoevsky_white_nights.txt

By adding --prob flag at the end, you can obtain:

* a dataframe with the probabilities per each author to have written a book, if you do not specify a test file;
* if you specify a test file, you will get the ordered list of authors, based on how likely they have written the book you requested. 

Whenever you do not specify a test file, you will only get the accuracy (unless you add --prob). Otherwise, you get the book information (title, author, predicted author). 

Also, you can switch from bag of words representation to tfidf by specifying --tfidf. 



## Open-ended project

For those who want to go further... The opposite of authorship attribution is obfuscation. You are Jane Austen and you don't want to have your texts identify you. What can you do to prevent this? Try out different methods and see if you can fool the system.

## Results

Leave One Out Mean Accuracies

Note that in case of the DecisionTreeClassifier performance are comparable with a random model that assigns a book to a random author (1/number of authors).


| **model_name**         | **accuracy** |
|------------------------|--------------|
| MultinomialNB          | 0.911111     |
| LinearSVC              | 0.877778     |
| LogisticRegression     | 0.844444     |
| RandomForestClassifier | 0.633333     |
| DecisionTreeClassifier | 0.077778     |

K fold with 5 splits

(mean)

| **model_name**         | **accuracy** |
|------------------------|--------------|
| LogisticRegression     | 0.844444     |
| MultinomialNB          | 0.811111     |
| LinearSVC              | 0.800000     |
| RandomForestClassifier | 0.577778     |
| DecisionTreeClassifier | 0.122222     |

(median)

| **model_name**         | **accuracy** |
|------------------------|--------------|
| MultinomialNB          | 0.833333     |
| LinearSVC              | 0.777778     |
| LogisticRegression     | 0.777778     |
| RandomForestClassifier | 0.555556     |
| DecisionTreeClassifier | 0.111111     |