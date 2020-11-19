import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import vocab as vb
import time
import math
import random
import unicodedata
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#读取数据
with open('./data/cmn.txt','r',encoding='utf-8') as f:
    data = f.readlines()
print('样本数：{}'.format(len(data)))

#分割英文数据和中文数据
en_data = [line.split('\t')[0] for line in data]
ch_data = [line.split('\t')[1] for line in data]

#按字符级切割，并添加<eos>
en_token_list = [[c for c in line] + ['<eos>'] for line in en_data]
ch_token_list = [[c for c in line] + ['<eos>'] for line in ch_data]

#基本字典
basic_dict = {'<pad>':0,'<unk>':1,'<bos>':2,'<eos>':3}
#分别生成中英文字典
en_vocab = set(''.join(en_data))
en2id = {c:i+len(basic_dict) for i,c in enumerate(en_vocab)}
en2id.update(basic_dict)
id2en = {v:k for k,v in en2id.items()}
#生成中文词典
ch_vocab = set(''.join(ch_data))
ch2id = {c:i+len(basic_dict) for i, c in enumerate(ch_vocab)}
ch2id.update(basic_dict)
id2ch = {v:k for k,v in ch2id.items()}

#利用字典，映射数据
en_num_data = [[en2id[en] for en in line] for line in en_token_list]
ch_num_data = [[ch2id[ch] for ch in line] for line in ch_token_list]

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

#表示为dataset
class TranslationDataset(Dataset):
    def __init__(self,src_data,trg_data):
        self.src_data = src_data
        self.trg_data = trg_data

        assert len(src_data) == len(trg_data),\
            "numbers of src_data and trg_data must be equal!"
    def __len__(self):
        return len(self.src_data)
    def __getitem__(self, idx):
        src_sample = self.src_data[idx]
        src_len = len(self.src_data[idx])
        trg_sample = self.trg_data[idx]
        trg_len = len(self.trg_data[idx])
        return {"src":src_sample,"src_len":src_len,"trg":trg_sample,"trg_len":trg_len}

def padding_batch(batch):
    """
    input: -> list of dict
        [{'src': [1, 2, 3], 'trg': [1, 2, 3]}, {'src': [1, 2, 2, 3], 'trg': [1, 2, 2, 3]}]
    output: -> dict of tensor
        {
            "src": [[1, 2, 3, 0], [1, 2, 2, 3]].T
            "trg": [[1, 2, 3, 0], [1, 2, 2, 3]].T
        }
    """
    en2id = vb.get_value('en2id')
    ch2id = vb.get_value('ch2id')
    src_lens = [d["src_len"] for d in batch]
    trg_lens = [d["trg_len"] for d in batch]
    src_max = max([d["src_len"] for d in batch])
    trg_max = max([d['trg_len'] for d in batch])
    for d in batch:
        d['src'].extend([en2id["<pad>"]] * (src_max - d["src_len"]))
        d["trg"].extend([ch2id["<pad>"]] * (trg_max - d["trg_len"]))
    srcs = torch.tensor([pair["src"] for pair in batch],dtype=torch.long,device=device)
    trgs = torch.tensor([pair["trg"] for pair in batch],dtype=torch.long,device=device)
    batch = {"src":srcs.T,"src_len":src_lens,"trg":trgs.T,"trg_len":trg_lens}
    return batch

