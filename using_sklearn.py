import pickle
from os import path
from sklearn.feature_extraction.text import HashingVectorizer as hv
from sklearn.naive_bayes import BernoulliNB as bnb
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score

# finds the file and opens it
file = path.abspath("sklearn-data.pickle")
data = pickle.load(open(file, "rb"))

# initialise
x_train, y_train, x_test, y_test = data["x_train"], data["y_train"], data["x_test"], data["y_test"]

# hashing
vector = hv(n_features=2**8, stop_words="english", binary=True)
x_train = vector.transform(x_train)
x_test = vector.transform(x_test)

# naive bayes classifier
nb_classifier = bnb()
nb_classifier.fit(x_train, y_train)
nb_pred_val = nb_classifier.predict(x_test)
nb_accuracy = accuracy_score(y_test, nb_pred_val)
print("The accuracy score for the Naive Bayes classifier is:", nb_accuracy)

# decision tree classifier
dt_classifier = dtc(max_features=10, criterion="gini")
dt_classifier.fit(x_train, y_train)
dt_pred_val = dt_classifier.predict(x_test)
dt_accuracy = accuracy_score(y_test, dt_pred_val)
print("The accuracy score for the Decision Tree classifier is:", dt_accuracy)





