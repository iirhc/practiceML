import load_data as load
import feature_list as flist
import model

if __name__ == '__main__':

    train_data = load.Data("tweets_SEMEVAL_train2013.txt")

    feature_list = flist.FeatureList()
    feature_list.add_dict("positive-words.txt")
    feature_list.add_dict("negative-words.txt")
    feature_list.add_feature(train_data)

    feature = feature_list.generate_feature(train_data)
    gold_standard = train_data.get_gold_standard()

    model = model.Model()
    model.train(feature, gold_standard)

    test_data = Data("tweets_SEMEVAL_test2013.txt")

    feature = feature_list.generate_feature(test_data)
    gold_standard = test_data.get_gold_standard()

    res = model.predict(feature, gold_standard)
    model.show()
