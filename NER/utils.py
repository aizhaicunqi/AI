def load_vocab(path):
    words = []
    for line in open(path):
        words.append(line.strip())
    word2id = {word: i for i, word in enumerate(words)}
    id2word = {i: word for i, word in enumerate(words)}
    return word2id, id2word


def char2id(chars, vocab):
    ids = []
    for char in chars:
        ids.append(vocab.get(char, vocab.get("UNK")))
    return ids


def pad_seq(seq, max_len, vocab):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [vocab.get("<PAD>", vocab.get("UNK"))] * (max_len - len(seq))


class DataProcess(object):
    def __init__(self, data_path, batch_size, max_len, word2id, tag2id):
        self.fp = open(data_path, "r")
        self.batch_size = batch_size
        self.max_len = max_len
        self.word2id = word2id
        self.tag2id = tag2id
        self.end = 0

    def get_batch(self):
        i = 0
        xs = []
        ys = []
        x = []
        y = []
        while i < self.batch_size:
            inp = self.fp.readline().strip()
            if inp == "":
                self.end = 1
                break
            if inp == "end":
                if len(y) > 0:
                    xs.append(pad_seq(x, self.max_len, self.word2id))
                    ys.append(pad_seq(y, self.max_len, self.tag2id))
                    x = []
                    y = []
                    i += 1
            else:
                inp_arr = inp.split("\t")
                x.append(self.word2id.get(inp_arr[0], "<UNK>"))
                y.append(self.tag2id.get(inp_arr[1], "<UNK>"))
        return xs, ys

    def get_all_data(self):
        xs = []
        ys = []
        x = []
        y = []
        while True:
            inp = self.fp.readline().strip()
            if inp == "":
                self.end = 1
                break
            if inp == "end":
                if len(y) > 0:
                    xs.append(x)
                    ys.append(y)
                    x = []
                    y = []
            else:
                inp_arr = inp.split("\t")
                x.append(inp_arr[0])
                y.append(inp_arr[1])
        return xs, ys
