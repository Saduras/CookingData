import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer

print("Reading data...")
train = json.load(open("./input/train.json"))
test = json.load(open("./input/test.json"))

# prepare data
labels = []
text_data = []
for recipe in train:
    text_data.append(" ".join(recipe["ingredients"]))
    if recipe["cuisine"] not in labels:
        labels.append(recipe["cuisine"])

for recipe in test:
    text_data.append(" ".join(recipe["ingredients"]))

join = " ".join(test[0]['ingredients'])

# create vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit( text_data )
# create map for labels
label_to_int = dict(((c, i) for i, c in enumerate(labels)))
int_to_label = dict(((i, c) for i, c in enumerate(labels)))

Y_train = []
X_train = []

for recipe in train:
    Y_train.append( label_to_int[recipe["cuisine"]] )
    X_train.append( vectorizer.transform( " ".join(recipe["ingredients"]) ))