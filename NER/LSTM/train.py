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

EMB_DIM = 128
HID_DIM = 128
DROPOUT = 0.2
LR = 0.001

model = NERLSTM(EMB_DIM, HID_DIM, DROPOUT, word2id, tag2id)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=LR)

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    # load word_lists, tag_lists
    dp = DataProcess("../data/train.txt", 64, 32, word2id, tag2id)
    index = 0
    while True:
        xs, ys = dp.get_batch()
        if dp.end == 1:
            break
        optimizer.zero_grad()
        ys = np.reshape(ys, newshape=(ys.shape[0] * ys.shape[1], ))
        # print(ys)
        pred = model(torch.LongTensor(xs))
        pred = pred.view(-1, pred.size(-1))
        # print(len(pred))
        loss = criterion(pred, torch.LongTensor(ys))
        loss.backward()
        optimizer.step()
        if index % 200 == 0:
            print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))
        index += 1

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
    print("-" * 25 + str(epoch) + "-" * 25)
    print(report)
    print("-" * 50)

torch.save(model, "./model.pkl")
