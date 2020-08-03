import pickle
from NER.utils import load_vocab, DataProcess
from NER.CRF.model import sent2features

# load vocab and tag_vocab
print("load vocab ...")
word2id, id2word = load_vocab("../data/vocab.txt")
tag2id, id2tag = load_vocab("../data/tag_vocab.txt")
print("tag2id:", tag2id)
print("load vocab done !!!")

# load word_lists, tag_lists
print("load data ...")
dp = DataProcess("../data/test.txt", 64, 32, word2id, tag2id)
word_lists, tag_lists = dp.get_all_data()
print("load data done")
print("data len: ", len(word_lists))


with open("./model.h5", "rb") as f:
    model = pickle.load(f)

features = [sent2features(s) for s in word_lists]
predicts = model.predict(features)
print(predicts)

result = []
for i in range(len(word_lists)):
    if len(word_lists[i]) != len(tag_lists[i]) != len(predicts[i]):
        print("error: ", i)
        continue
    for j in range(len(word_lists[i])):
        result.append("%s\t%s\t%s" % (word_lists[i][j], tag_lists[i][j], predicts[i][j]))

import os
result_file = "./output_predict.txt"
if os.path.exists(result_file):
    os.remove(result_file)
result_fp = open(result_file, "a+")
for res in result:
    result_fp.write("%s\n" % res)
result_fp.close()

