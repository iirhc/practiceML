from load_data import Data
from feature_list import FeatureList
from model import Model

if __name__ == '__main__':

    train_data = Data("tweets_SEMEVAL_train2013.txt")

    feature_list = FeatureList()
    feature_list.add_dict("positive-words.txt")
    feature_list.add_dict("negative-words.txt")
    feature_list.add_feature(train_data.get_tokens())

    feature = feature_list.generate_feature(train_data.get_tokens())
    gold_standard = train_data.get_gold_standard()

    model = Model()
    model.train(feature, gold_standard)

    test_data = Data("tweets_SEMEVAL_test2013.txt")

    feature = feature_list.generate_feature(test_data.get_tokens())
    gold_standard = test_data.get_gold_standard()

    res = model.predict(feature, gold_standard)
    model.show()
