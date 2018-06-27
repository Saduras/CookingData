import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import seaborn as sns
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from time import time
start = time()

print("Reading data...")
train = json.load(open("./input/train.json"))
test = json.load(open("./input/test.json"))

print("Vectorize data...")
labels = []
text_data = []
for recipe in train:
    text_data.append(" ".join(recipe["ingredients"]))
    if recipe["cuisine"] not in labels:
        labels.append(recipe["cuisine"])

# for recipe in test:
#     text_data.append(" ".join(recipe["ingredients"]))

join = " ".join(test[0]['ingredients'])

# create vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit( text_data )
# create map for labels
label_to_int = dict(((c, i) for i, c in enumerate(labels)))
int_to_label = dict(((i, c) for i, c in enumerate(labels)))

Y_train = [ label_to_int[r["cuisine"]] for r in train ]
X_train = vectorizer.transform( text_data )

print("Fit classifier...")
classifier_names = ["SVC", "DecisionTree", "AdaBoost", "RandomForest", "ExtraTrees", "GradientBoosting", "MultipleLayerPerceptron", "KNeighbors", "LogisticRegression", "LinearDiscriminantAnalysis"]

random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state), random_state=random_state, learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state=random_state))
classifiers.append(LinearDiscriminantAnalysis())

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)

cv_results = []
for i, classifier in enumerate(classifiers):
    print(f"Fitting {classifier_names[i]} now...")
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs = 4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means, "CrossValErrors":cv_std, "Algorithm":classifier_names})
cv_res.to_csv("./cv_res.csv")

g = sns.barplot(x="CrossValMeans", y="Algorithm", data = cv_res, palette="Set3", orient = "h", **{'xerr': cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")

print(f"duration: {time() - start:.2}s")