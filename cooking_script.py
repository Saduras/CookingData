import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=0.1)
ada.fit(X_train, Y_train)

pred = ada.predict(X_train)
score = accuracy_score(Y_train, pred)
print(score)
print(f"duration: {time() - start:.2}s")