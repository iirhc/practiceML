import re
import numpy as np
import pickle
import time
from sklearn.svm import SVC
from itertools import islice  # to ignore first line

file1 = open("tweets_SEMEVAL_train2013.txt")

patten = "[\w'-]+"
collect = []
feature_set = set()

i = 0
for line in islice(file1, 1, None):
    match = re.findall(patten, line)
    match.pop(0)
    last = int(match.pop())
    aline = [match, last]
    collect.append(aline)
    for m in match:
        feature_set.add(m)
feature_list = list(feature_set)
feature_list.sort()
print("load training data finished")
print(len(feature_list))

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

clf = SVC(C=100)
tstart = time.time()
clf.fit(feature, gold_standard)
tend = time.time()
print("SVC finished")
print(tend - tstart)

# save the classifier
with open('feature_list.pkl', 'wb') as f:
    pickle.dump(feature_list, f)
with open('classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("save classifier finished")
