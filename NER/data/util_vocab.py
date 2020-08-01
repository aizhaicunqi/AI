import os


def build_vocab(corpus_file_list, vocab_file, tag_file):
    words = set()
    tags = set()
    for file in corpus_file_list:
        for line in open(file, "r", encoding='utf-8').readlines():
            line = line.strip()
            if line == "end":
                continue
            try:
                w, t = line.split()
                words.add(w)
                tags.add(t)
            except Exception as e:
                print(line.split())
                # raise e

    if not os.path.exists(vocab_file):
        with open(vocab_file, "w") as f:
            for index, word in enumerate(["<PAD>", "<UKN>"] + list(words)):
                f.write(word + "\n")

    tag_sort = {
        "O": 0,
        "B": 1,
        "I": 2,
        "E": 3,
    }

    tags = sorted(list(tags),
                  key=lambda x: (len(x.split("-")), x.split("-")[-1], tag_sort.get(x.split("-")[0], 100))
                  )
    if not os.path.exists(tag_file):
        with open(tag_file, "w") as f:
            for index, tag in enumerate(["<PAD>", "<UKN>"] + tags):
                f.write(tag + "\n")


build_vocab(["./train.txt", "./test.txt"], "./vocab.txt", "./tag_vocab.txt")
