import re
import pickle
import time
from itertools import islice

class FeatureList:
    def __init__(self):
        self.feature_list = []
        self.dictionary_list = []

    def add_dict(self, path, ignore=35):
        sentiment_dict = open(path)
        patten = "[\w'-]+"
        dict_list = []
        for line in islice(sentiment_dict, ignore, None):
            match = re.findall(patten, line)
            dict_list.append(match)
        self.dictionary_list += dict_list
        print("import sentiment dictionary finished.")

    def add_feature(self, data):
        feature_set = data.get_word_set()
        self.feature_list += list(feature_set)
        print("import data feature finished.")

    def add_feature_from_save(self, path="temp/data.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        feature_set = set()
        for datum in data:
            for word in datum[0]:
                feature_set.add(word)
        self.feature_list += list(feature_set)
        print("import data feature finished.")

    def generate_feature(self, data):
        print(len(self.feature_list)+len(self.dictionary_list))
        data = data.get_tweets()
        features = []
        tstart = time.time()
        for datum in data:
            content = datum.get_content()
            feature = []
            for word in self.feature_list:
                if word in content:
                    # append TFIDF here
                    feature.append(datum.get_tfidf(word))
                else:
                    feature.append(0)
            for word in self.dictionary_list:
                if word in content:
                    feature.append(1)
                else:
                    feature.append(0)
            features.append(feature)
        tend = time.time()
        print("generate feature finished, spent time: ", end="")
        print(tend - tstart)
        return features

    def save(self, file_name="feature_list"):
        with open("temp/"+file_name+".pkl", "wb") as f:
            pickle.dump(self.feature_list, f)

    def load(self, file_name="feature_list"):
        with open("temp/"+file_name+".pkl", "rb") as f:
            self.feature_list = pickle.load(f)
