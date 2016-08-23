import re
import pickle
from itertools import islice  # to ignore header

class Data:
    def __init__(self, path, patten="[\w'-]+"):
        self.raw = open(path)
        self.collect = []
        self.tokens = []
        self.gold_standard = []
        self.parse(patten, 1)
        #self.save()
        print("load data finished.")

    def parse(self, patten, ignore):
        for line in islice(self.raw, ignore, None):
            match = re.findall(patten, line)
            match.pop(0)
            label = match.pop()
            self.collect.append([match, label])
            self.tokens.append(match)
            self.gold_standard.append(label)

    def save(self, file_name="data"):
        with open("temp/"+file_name+".pkl", "wb") as f:
            pickle.dump(self.collect, f)

    def get_data(self):
        return self.collect

    def get_tokens(self):
        return self.tokens

    def get_gold_standard(self):
        return self.gold_standard
