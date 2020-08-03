from sklearn_crfsuite import CRF  # CRF的具体实现太过复杂，这里我们借助一个外部的库
import pickle


def word2features(sent, i):
    """抽取单个字的特征"""
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i - 1]
    next_word = "</s>" if i == (len(sent) - 1) else sent[i + 1]
    # 因为每个词相邻的词会影响这个词的标记
    # 所以我们使用：
    # 前一个词，当前词，后一个词，
    # 前一个词+当前词， 当前词+后一个词
    # 作为特征
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word + word,
        'w:w+1': word + next_word,
        'bias': 1
    }
    return features


def sent2features(sent):
    """抽取序列特征"""
    return [word2features(sent, i) for i in range(len(sent))]


class CRFModel(object):
    def __init__(self,
                 algorithm='lbfgs',
                 c1=0.1,
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False,
                 save_model_path="./model.h5"
                 ):
        self.save_model_path = save_model_path
        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self, sentences, tag_lists):
        """训练模型"""
        features = [sent2features(s) for s in sentences]
        self.model.fit(features, tag_lists)

    def save(self):
        with open(self.save_model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def test(self, sentences):
        """解码,对给定句子预测其标注"""
        with open(self.save_model_path, 'rb') as f:
            self.model = pickle.load(f)
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists
