import re
import numpy as np
from itertools import islice # to ignore first line

file = open("tweets_SEMEVAL_train2013.txt")

patten = "[\w']+"
collect = []
feature_set = set()

for line in islice(file, 1, None):
    match = re.findall(patten, line)
    match.pop(0)
    last = int(match.pop())
    aline = [match, last]
    collect.append(aline)
    for m in match:
        feature_set.add(m)
