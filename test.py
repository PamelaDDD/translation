from model import *
from config import *
from torch import optim
from dataset import normalizeString
seed = 2000
def translate(model,sample,idx2token=None):
    model.predict = True
    model.eval()
    #shape -> [seq_len,1]
    input_batch = sample["src"]
    #list
    input_len = sample["src_len"]
    output_tokens = model(input_batch,input_len)
    output_tokens = [idx2token[t] for t in output_tokens]
    return "".join(output_tokens)

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

    # 按词级切割，并添加<eos>
    en_token_list = [[c for c in line.split(' ') if c != ''] + ['<eos>'] for line in en_data]
    ch_token_list = [[c for c in line.split(' ') if c != ''] + ['<eos>'] for line in ch_data]
    print('英文数据:\n', en_token_list[:2])
    print('\n中文数据:\n', ch_token_list[:2])

    # 基本字典
    basic_dict = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
    # 分别生成英文字典
    en_vocab = set()
    for line in en_data:
        for word in line.split(' '):
            en_vocab.add(word)
    en2id = {c: i + len(basic_dict) for i, c in enumerate(en_vocab)}
    en2id.update(basic_dict)
    id2en = {v: k for k, v in en2id.items()}


    # 生成中文词典
    ch_vocab = set()
    for line in ch_data:
        for word in line.split(' '):
            ch_vocab.add(word)
    ch2id = {c: i + len(basic_dict) for i, c in enumerate(ch_vocab)}
    ch2id.update(basic_dict)
    id2ch = {v: k for k, v in ch2id.items()}

    # 利用字典，映射数据
    en_num_data = [[en2id[en] for en in line] for line in en_token_list]
    ch_num_data = [[ch2id[ch] for ch in line] for line in ch_token_list]

    print("char:",ch_data[1])
    print("index:",ch_num_data[1])



    INPUT_DIM = len(ch2id)
    OUTPUT_DIM = len(en2id)

    #加载最优权重
    for k,v in basic_dict.items():
        print(k,v)
    bidirectional = True
    enc = Encoder(INPUT_DIM,ENC_EMB_DIM,HID_DIM,N_LAYERS,ENC_DROPOUT,bidirectional)
    dec = Decoder(OUTPUT_DIM,DEC_EMB_DIM,HID_DIM,N_LAYERS,DEC_DROPOUT,bidirectional)
    model = Seq2seq(enc,dec,device,basic_dict=basic_dict).to(device)

    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
    model.load_state_dict(torch.load('ch2en-model.pt'))
    random.seed(seed)
    for i in random.sample(range(len(ch_num_data)),10):  #随机看10个
        ch_tokens = list(filter(lambda  x : x!=0,ch_num_data[i]))
        en_tokens = list(filter(lambda  x : x!=3 and x != 0, en_num_data[i]))
        sentence = [id2ch[t] for t in ch_tokens]
        print("【原文】")
        print("".join(sentence))
        translation = [id2en[t] for t in en_tokens]
        print("【ground true】")
        print("".join(translation))
        test_sample = {}
        test_sample["src"] = torch.tensor(ch_tokens,dtype=torch.long,device=device).reshape(-1,1)
        test_sample["src_len"] = [len(ch_tokens)]
        print("【机器翻译】")
        print(translate(model,test_sample,id2en),end="\n\n")