import pandas as pd
import json

print("Reading data...")
train = json.load(open("./input/train.json"))
test = json.load(open("./input/test.json"))