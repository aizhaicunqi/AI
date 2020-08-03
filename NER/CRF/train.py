from NER.CRF.model import CRFModel
from NER.utils import load_vocab, DataProcess

# load vocab and tag_vocab
word2id, id2word = load_vocab("../data/vocab.txt")
tag2id, id2tag = load_vocab("../data/tag_vocab.txt")
print("tag2id:", tag2id)

# load word_lists, tag_lists
dp = DataProcess("../data/train.txt", 64, 32, word2id, tag2id)
word_lists, tag_lists = dp.get_all_data()

model = CRFModel()
print("training ...")
model.train(word_lists, tag_lists)
print("training done !!!")
model.save()
print("save model done !!!")

