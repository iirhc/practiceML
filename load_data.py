import re
import pickle
from itertools import islice  # to ignore header
from tweet import Tweet
from math import log10

class Data:
    def __init__(self, path, patten="[\w'-]+"):
        self.raw = open(path)
        self.parse(patten, 1)
        #self.save()
        print("load data finished.")

    def parse(self, patten, ignore):
        self.tweets = []
        tweet_num = 0
        self.word_set = set()
        self.idf = {}
        for line in islice(self.raw, ignore, None):
            match = re.findall(patten, line)
            match.pop(0)
            label = match.pop()
            self.tweets.append(Tweet([match, label]))
            tokens = set()
            for word in match:
                tokens.add(word)
                self.word_set.add(word)
            for word in tokens:
                if self.idf.setdefault(word, 0)!=None:
                    self.idf[word] += 1
            tweet_num += 1
        for word in self.idf:
            self.idf[word] = log10(tweet_num/self.idf[word])
        for tweet in self.tweets:
            tweet.calc_tfidf(self.idf)

    def save(self, file_name="data"):
        with open("temp/"+file_name+".pkl", "wb") as f:
            pickle.dump(self.collect, f)

    def get_data(self):
        return self.collect

    def get_tweets(self):
        return self.tweets

    def get_gold_standard(self):
        gold_standard = []
        for tweet in self.tweets:
            gold_standard.append(tweet.get_gold_standard())
        return gold_standard

    def get_word_set(self):
        return self.word_set
