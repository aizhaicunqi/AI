import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from NER.HMM.model import HMM
from NER.utils import load_vocab, DataProcess

# load vocab and tag_vocab
word2id, id2word = load_vocab("../data/vocab.txt")
tag2id, id2tag = load_vocab("../data/tag_vocab.txt")
print("tag2id:", tag2id)

# load word_lists, tag_lists
dp = DataProcess("../data/train.txt", 64, 32, word2id, tag2id)
word_lists, tag_lists = dp.get_all_data()

hmm = HMM(len(tag2id), len(word2id))
hmm.train(word_lists, tag_lists, word2id, tag2id)
