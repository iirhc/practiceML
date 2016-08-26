class Tweet:
    def __init__(self, data):
        self.content = data[0]
        self.gold_standard = data[1]
        self.calc_tf()

    def calc_tf(self):
        self.tf = {}
        word_num = len(self.content)
        word_dict = {}
        for word in self.content:
            if word_dict.setdefault(word, 0)!=None:
                word_dict[word] += 1
        for word in word_dict:
            self.tf[word] = word_dict[word]/word_num

    def calc_tfidf(self, idf_set):
        self.tfidf = {}
        for word in self.tf:
            self.tfidf[word] = self.tf[word] * idf_set[word]

    def get_content(self):
        return self.content

    def get_gold_standard(self):
        return self.gold_standard

    def get_tfidf(self, word):
        return self.tfidf[word]

    def get_tfidf_set(self):
        return self.tfidf
