#%%
################################################
# K-FOLD CROSS VALIDATION 
# WITH ALL THE PROPOSED MODELS
################################################
import warnings
from utils import *
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from termcolor import colored

warnings.filterwarnings("ignore")


#########################################
# Preparing data for the cross validation
#########################################

# Read authors and texts
print(colored("Reading all texts...",'blue'))
texts, authors = get_authors_and_texts(TEXT_PATH)


# Creation of the associated labels
print(colored("Creating authors labels...",'blue'))
classes, author_dict = create_labels_vector(authors, AUTHORS_INFORMATION_PATH)

# Bag of words representation
print(colored("Representing the text as vectors...",'blue'))
vectorizer, vectors = generate_text_vectors(texts,'bow')


# To get n-1 in training set and 1 in test set
# Note that it requires 30 min to execute
# cv = LeaveOneOut()

cv = KFold(n_splits=5, shuffle=True, random_state=0)
models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    DecisionTreeClassifier(max_depth=2),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

classes, author_dict = create_labels_vector(authors, AUTHORS_INFORMATION_PATH)
# Bag of words representation
vectors = vectorizer.fit_transform(texts)

entries = []
from tqdm import tqdm

for model in tqdm(models):
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, vectors, classes, cv=cv, n_jobs=1)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=["model_name", "fold_idx", "accuracy"])
#%%
# Mean results
cv_df.groupby(["model_name"]).mean().sort_values("accuracy", ascending=False) 
# Median results
cv_df.groupby(["model_name"]).median().sort_values("accuracy", ascending=False)
