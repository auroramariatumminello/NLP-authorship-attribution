# 📚 Authorship attribution


## 🎯 Goal of the project
This repository shows how it is possible to discover the author of a text book, given a list of authors among which we can choose. It may be resused for plagiarism detection or to discover the author of messages, by changing the list of authors and the directory with txt documents necessary for the training phase. 

The model will try to guess the true author of the text you choose (or some random ones the model has never seen taken from the directory of predownloaded books). 

## 🤓 Requirements

If you want to execute the code, it is suggested to create a conda environment where to install the [requirements](requirements.txt): 

    conda create --name your_env 
    conda activate your_env

    conda install pip # Optional for those with no python installation
    pip install -r requirements.txt


If you're new to NLTK library, you should also run: 

    nltk.download("punkt")
    nltk.download("stopwords")

## 🐍 Run the code

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






## 📁Structure of the repo

The repository is divided into different folders:

* data, where you can find three different dataset for training ad testing the code (American/British, Russian and Italian literature). Notice that russian and italian folders have few books because of the lack of literature on Gutenberg.
* images, with some heatmaps that serve as example of the jupyter notebooks and the accuracy of different models;
* original_setup, which contains the original code (with some improvements) in the repo [Authorship Attribution](https://github.com/ml-for-nlp/authorship-attribution) of Natural Language Processing course at University of Trento.

Additional files can be found in the main directory:

* attribution.ipynb is a notebook that guides you throughout the entire project, allowing you to customize it in a user friendly way;
* attribution.py is a python script to execute the code from the command line, such that you can request either the accuracy score based on a specific training path or the prediction of a particular test book;
* utils.py with some utilities function to execute the code;
* cross_validation.py, necessary in the Results section to choose the best model for this use case;
* requirements.txt to replicate the same environment to execute the code.
   

## 🔢 Results

Multiple models are suggested in order to classify a text, but mainly MultinomialNB is used. Why?

*The following results are computed on the American/British data, which is the biggest folder available.*

### Leave One Out Cross Validation

If we run a [cross validation](cross_validation.py) with the leave one out k-fold, this means that the model will train on $n-1$ texts, leaving $1$ to testing. The mean accuracies are showed in the table below, shoing that MultinomialNB is the best model, followed by LinearSVC and LogisticRegression.

> Note that in case of the DecisionTreeClassifier performance are comparable with a random model that assigns a book to a random author (1/number of authors).


| **model_name**         | **mean accuracy** |
|------------------------|--------------|
| MultinomialNB          | 0.911111     |
| LinearSVC              | 0.877778     |
| LogisticRegression     | 0.844444     |
| RandomForestClassifier | 0.633333     |
| DecisionTreeClassifier | 0.077778     |

### K fold Cross Validation

The cross validation script also presents a proper K-fold cross validation with 5 splits, which means that it tries 5 different configurations of training/testing sets. You can find the results below, considered as mean and median accuracy. If we consider the mean accuracy, LogisticRegression performs slightly better than MultinomialNB, which instead is the best model when looking at the median accuracy. 


| **model_name**         | **mean accuracy** |**median accuracy** |
|------------------------|-------------------|--------------------|
| LogisticRegression     | 0.844444          |0.777778            |
| MultinomialNB          | 0.811111          |0.833333            |
| LinearSVC              | 0.800000          |0.777778            |
| RandomForestClassifier | 0.577778          |0.555556            |
| DecisionTreeClassifier | 0.122222          |0.111111            |



## 📖 Test your bookshelf

Suppose you want to predict the author of random books from a range of authors you'd like to choose:

1. Download the books from [https://www.gutenberg.org/ebooks/](https://www.gutenberg.org/ebooks/) and name them with the format ``uthor_surname_title_of_your_book.txt`. Remember to download at least 2 books per author!
2. Insert them all inside a new directory
3. Insert the test books inside a folder test_books
4. Create a csv named `authors.csv` with the columns name and label, which represent the complete name of the author and the surname used to name book files
5. Specify the new training path to the script (and inside the jupyter notebook if you're using it) like this:
    
        python attribution.py --training_path <new-path> --test_fiile <test_book> --prob


>⚠️*Attention! The basic algorithm works for all languages, but if you choose to recur to tfidf instead of bag of words, NLTK does not work with other languages, therefore the code should be adapted to your personal language through spacy.*