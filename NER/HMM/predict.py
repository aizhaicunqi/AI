import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from NER.HMM.model import HMM
from NER.utils import load_vocab, DataProcess

# load vocab and tag_vocab
word2id, id2word = load_vocab("../data/vocab.txt")
tag2id, id2tag = load_vocab("../data/tag_vocab.txt")
print("tag2id:", tag2id)

hmm = HMM(len(tag2id), len(word2id))

# load word_lists, tag_lists
dp = DataProcess("../data/test.txt", 64, 32, word2id, tag2id)
word_lists, tag_lists = dp.get_all_data()


preds, labels = [], []
total = len(tag_lists[:1000])
for i in range(total):
    word_list = word_lists[i]
    tag_list = tag_lists[i]
    pre_tag_list = hmm.decoding(word_list, word2id, tag2id)
    preds.extend(pre_tag_list)
    labels.extend(tag_list)
    if i % 100 == 0:
        print("已预测：%s, 总数：%s" % (i, total))

precision = precision_score(labels, preds, average='macro')
recall = recall_score(labels, preds, average='macro')
f1 = f1_score(labels, preds, average='macro')
report = classification_report(labels, preds)
print(report)
