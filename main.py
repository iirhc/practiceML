import re
import numpy as np
import pickle
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from itertools import islice  # to ignore first line

with open('feature_list.pkl', 'rb') as f:
    feature_list = pickle.load(f)
with open('classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
print("load classifier finished")

file2 = open("tweets_SEMEVAL_test2013.txt")

patten = "[\w'-]+"
collect = []

i = 0
for line in islice(file2, 1, None):
    match = re.findall(patten, line)
    match.pop(0)
    last = int(match.pop())
    aline = [match, last]
    collect.append(aline)
print("load testing data finished")

feature = []
gold_standard = []

for tweet in collect:
    tweet_feature = []
    for f in feature_list:
        if f in tweet[0]:
            tweet_feature.append(1)
        else:
            tweet_feature.append(0)
    feature.append(tweet_feature)
    gold_standard.append(tweet[1])
print("get feature finished")

feature = np.array(feature)
gold_standard = np.array(gold_standard)
print("numpy finished")

tstart = time.time()
res = clf.predict(feature)
tend = time.time()
print(accuracy_score(gold_standard, res))
print(classification_report(gold_standard, res, [1, 0, -1]))
print(tend - tstart)
