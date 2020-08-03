import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from NER.LSTM.model import NERLSTM
from NER.utils import load_vocab, DataProcess
from torch import nn, optim
import numpy as np
import torch

# load vocab and tag_vocab
word2id, id2word = load_vocab("../data/vocab.txt")
tag2id, id2tag = load_vocab("../data/tag_vocab.txt")
print("tag2id:", tag2id)

model = torch.load("./model.pkl")
model.eval()
# valid
model.eval()
dp_test = DataProcess("../data/test.txt", 64, 32, word2id, tag2id)
preds, labels = [], []
while True:
    test_xs, test_ys = dp_test.get_batch()
    if dp_test.end == 1:
        break
    # print(test_xs.shape)
    predict = model(torch.LongTensor(test_xs))
    predict = torch.argmax(predict, dim=-1)
    predict = predict.tolist()
    for pred in predict:
        pred = [id2tag[p] for p in pred]
        preds.extend(pred)
    for test_y in test_ys:
        test_y = [id2tag[p] for p in test_y]
        labels.extend(test_y)

precision = precision_score(labels, preds, average='macro')
recall = recall_score(labels, preds, average='macro')
f1 = f1_score(labels, preds, average='macro')
report = classification_report(labels, preds)
print(report)

import os
result_file = "./output.txt"
if os.path.exists(result_file):
    os.remove(result_file)
result_fp = open(result_file, "a+")
result_fp.write("%s\n" % report)
