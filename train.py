import torch
import time
from model import Encoder, Decoder, Seq2seq
from config import *
import torch.optim as optim
from dataset import TranslationDataset, padding_batch
from torch.utils.data import Dataset, DataLoader
from tools import showPlot
from vocab import *
import vocab as vb
from dataset import normalizeString

vb._init()
import os
import torch

device_ids = [1, 2, 3, 4]


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, data_loader, optimizer, clip=1, teacher_forcing_ratio=0.5, print_every=None):  # None 不打印
    model.predict = False
    model.train()

    if print_every == 0:
        print_every = 1
    print_loss_total = 0  # 每次打印都重置
    start = time.time()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        # shape -> [seq_len,batch]
        input_batches = batch["src"]
        target_batches = batch["trg"]

        # list
        input_lens = batch["src_len"]
        target_lens = batch["trg_len"]

        optimizer.zero_grad()

        loss = model(input_batches, input_lens, target_batches, target_lens, teacher_forcing_ratio)
        print_loss_total += loss.item()
        epoch_loss += loss.item()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optimizer.step()

        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\t Current loss : %4f' % print_loss_avg)
    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, print_every=None):
    model.predict = False
    model.eval()
    if print_every == 0:
        print_every = 1
    print_loss_total = 0  # 每次打印都重置
    start = time.time()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # shape -> [seq_len,batch]
            input_batches = batch["src"]
            target_batches = batch["trg"]
            # list
            input_lens = batch["src_len"]
            target_lens = batch["trg_len"]

            loss = model(input_batches, input_lens, target_batches, target_lens, teacher_forcing_ratio=0)
            print_loss_total += loss.item()
            epoch_loss += loss.item()
            if print_every and (i + 1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('\t Current Loss:%4f' % print_loss_avg)
    return epoch_loss / len(data_loader)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    vb.set_value('en2id', en2id)
    vb.set_value('id2en', id2en)
    vb.set_value('ch2id', ch2id)
    vb.set_value('id2ch', id2ch)

    INPUT_DIM = len(ch2id)
    OUTPUT_DIM = len(en2id)

    bidirectional = True
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, bidirectional)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, bidirectional)
    model = Seq2seq(enc, dec, device, basic_dict=basic_dict).to(device)
    # encoder 和 decoder 设置相同的学习策略
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 数据集
    train_set = TranslationDataset(ch_num_data, en_num_data)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=padding_batch,
                              shuffle=True)

    best_valid_loss = float('inf')
    epoch_train_losses = []
    epoch_val_losses = []
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, CLIP)
        vaild_loss = evaluate(model, train_loader)
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(vaild_loss)
        end_time = time.time()

        if vaild_loss < best_valid_loss:
            best_valid_loss = vaild_loss
            torch.save(model.state_dict(), 'ch2en-model.pt')
        if epoch % 2 == 0:
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch:{epoch + 1:02} | Time:{epoch_mins}m {epoch_secs}s')
            print(f'\tTrain loss: {train_loss:.3f} | Val.Loss:{vaild_loss:.3f}')
    print("best valid loss:", best_valid_loss)
    showPlot(epoch_train_losses, './result/train_loss.png')
    showPlot(epoch_val_losses, './result/val_loss.png')

    model.load_state_dict(torch.load('ch2en-model-word.pt'))
