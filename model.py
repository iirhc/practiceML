import numpy as np
import pickle
import time
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class Model:
    def __init__(self):
        self.clf = LinearSVC(C=100)

    def train(self, feature, gold_standard):
        tstart = time.time()
        self.clf.fit(np.array(feature), np.array(gold_standard))
        tend = time.time()
        self.save()
        print("LinearSVC train finished, spent time: ", end="")
        print(tend - tstart)

    def predict(self, feature, gold_standard):
        tstart = time.time()
        self.res = self.clf.predict(np.array(feature))
        tend = time.time()
        self.gold_standard = np.array(gold_standard)
        print("LinearSVC predict finished, spent time: ", end="")
        print(tend - tstart)

    def show(self):
        print("accuracy score: ", end="")
        print(accuracy_score(self.gold_standard, self.res))
        print("classification_report: ")
        print(classification_report(self.gold_standard, self.res))

    def save(self, file_name="model"):
        with open("temp/"+file_name+".pkl", "wb") as f:
            pickle.dump(self.clf, f)

    def load(self, file_name="model"):
        with open("temp/"+file_name+".pkl", "wb") as f:
            self.clf = pickle.load(f)
