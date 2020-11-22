import pickle
import unicodedata
import re

basic_dict = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?]|[，。？！])", r" ", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


class Vocab:
    def __init__(self,name):
        self.name = name
        self.word2id = {}
        self.id2word = {}
        self.word_num_data = []

    def gen_token_list(self,sentences):
        self.token_list = [[c for c in line.split(' ') if c != ''] + ['<eos>'] for line in sentences]

    def gen_vocab_dict(self,sentences):
        vocab = set()
        for line in sentences:
            for word in line.split(' '):
                vocab.add(word)
        word2id = {c: i + len(basic_dict) for i, c in enumerate(vocab)}
        word2id.update(basic_dict)
        id2word = {v: k for k, v in word2id.items()}
        self.word2id = word2id
        self.id2word = id2word

    def gen_word_num_data(self,sentences):
        self.gen_token_list(sentences)
        self.gen_vocab_dict(sentences)
        self.word_num_data = [[self.word2id[c] for c in line] for line in self.token_list]

    def get_name(self):
        return self.name

    def get_word2id(self):
        return self.word2id

    def get_id2word(self):
        return self.id2word

    def get_word_num_data(self):
        return self.word_num_data


if __name__ == '__main__':
    # 读取数据
    data = []
    with open('./data/en-ch_word.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = normalizeString(line.strip())
            data.append(line)

    # 分割英文数据和中文数据
    en_data = [line.lower().split('\t')[0] for line in data]
    ch_data = [line.split('\t')[1] for line in data]
    print('英文数据:\n', en_data[:10])
    print('中文数据:\n', ch_data[:10])

    CHVOCAB = Vocab('ch')
    CHVOCAB.gen_word_num_data(ch_data)
    ch2id = CHVOCAB.get_word2id()
    id2ch = CHVOCAB.get_id2word()
    ch_num_data = CHVOCAB.get_word_num_data()

    ENVOCAB = Vocab('en')
    ENVOCAB.gen_word_num_data(en_data)
    en2id = ENVOCAB.get_word2id()
    id2en = ENVOCAB.get_id2word()
    en_num_data = ENVOCAB.get_word_num_data()


    with open('ch2id.pickle','wb') as handle:
        pickle.dump(ch2id,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open('id2ch.pickle','wb') as handle:
        pickle.dump(id2ch,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open('ch_num_data.pickle','wb') as handle:
        pickle.dump(ch_num_data,handle,protocol=pickle.HIGHEST_PROTOCOL)



    with open('en2id.pickle','wb') as handle:
        pickle.dump(en2id,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open('id2en.pickle','wb') as handle:
        pickle.dump(id2en,handle,protocol=pickle.HIGHEST_PROTOCOL)

    with open('en_num_data.pickle','wb') as handle:
        pickle.dump(en_num_data,handle,protocol=pickle.HIGHEST_PROTOCOL)

    #Load data
    with open('ch2id.pickle','rb') as handle:
        b = pickle.load(handle)

    with open('basic_dict.pickle','wb') as handle:
        pickle.dump(basic_dict,handle,protocol=pickle.HIGHEST_PROTOCOL)
