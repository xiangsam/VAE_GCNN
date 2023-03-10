'''
Author: Samrito
Date: 2023-03-09 19:11:29
LastEditors: Samrito
LastEditTime: 2023-03-10 21:35:19
'''
import re
import codecs
import numpy as np
import sys
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchsummaryX import summary
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader


class ShiDataset(Dataset):
    def __init__(self, datafile):
        super().__init__()
        self.n = 5  # 只抽取5言诗
        s = codecs.open(datafile, encoding='utf-8').read()
        s = re.findall(u'　　(.{%s}，.{%s}。.*?)\r\n' % (self.n, self.n), s)
        shi = []
        for i in s:
            for j in i.split(u'。'):  # 按句切分
                if j:
                    shi.append(j)
        shi = [
            i[:self.n] + i[self.n + 1:] for i in shi
            if len(i) == 2 * self.n + 1
        ]
        self.id2char = dict(enumerate(set(''.join(shi))))
        self.char2id = {j: i for i, j in self.id2char.items()}
        self.shi2id = np.array([[self.char2id[j] for j in i] for i in shi])

    def __len__(self):
        return self.shi2id.shape[0]

    def __getitem__(self, index):
        return self.shi2id[index]


class GCNN(nn.Module):
    def __init__(self, seq_len, emb_dim, residual=False):
        super(GCNN, self).__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.residual = residual
        self.conv = nn.Conv2d(1, emb_dim * 2, (3, emb_dim), padding=(1, 0))
        self.b = nn.parameter.Parameter(torch.randn(1, emb_dim * 2, seq_len,
                                                    1))
        nn.init.xavier_uniform_(self.b)

    def forward(self, x):
        # x: (bs, seq_len, emb_dim)
        _ = x.unsqueeze(1)  # (bs, 1, seq_len, emb_dim)
        h = (self.conv(_) + self.b).squeeze(-1)  #h: (bs, emb_dim*2, seq_len)
        A = h[:, :self.emb_dim, :]
        B = h[:, self.emb_dim:, :]
        A = A.permute(0, 2, 1)
        B = B.permute(0, 2, 1)
        out = A * torch.sigmoid(B)
        if self.residual:
            return x + out
        return out


class VAE(nn.Module):
    def __init__(self,
                 seq_len,
                 vocab_size,
                 emb_dim,
                 hidden_dim=None,
                 residual=False):
        super(VAE, self).__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        if hidden_dim is None:
            hidden_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.gcnn1 = GCNN(seq_len, emb_dim, residual)
        self.gcnn2 = GCNN(seq_len, emb_dim, residual)
        self.gcnn3 = GCNN(seq_len, emb_dim, residual)
        self.fc11 = nn.Linear(emb_dim, self.hidden_dim)
        self.fc12 = nn.Linear(emb_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.emb_dim * self.seq_len)
        self.fc3 = nn.Linear(self.emb_dim, vocab_size)

    def reparameterize(self, mu, logvar):
        '''
        采样不可导，但是采样结果可导。直接通过重参数技巧得到采样结果，避免采样过程参与梯度下降
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.normal(mean=0, std=1,
                           size=(mu.shape[0], self.hidden_dim)).cuda()
        return mu + eps * std

    def encode(self, x):
        #x: (bs, seq_len)
        emb = self.emb(x)  #emb: (bs, seq_len, emb_dim)
        h = self.gcnn1(emb)
        h = self.gcnn2(emb)  #h: (bs, seq_len, emb_dim)
        h = h.mean(1)  #h: (bs, emb_dim)
        mu = self.fc11(h)  # mu: (bs, hidden_dim)
        log_var = self.fc12(h)

        return mu, log_var

    def decode(self, z):
        #z: (bs, hidden_dim)
        h = self.fc2(z)  #h: (bs, emb_dim * seq_len)
        h = h.reshape(-1, self.seq_len,
                      self.emb_dim)  # h: (bs, seq_len, emb_dim)
        h = self.gcnn3(h)  #h: (bs, seq_len, emb_dim)
        h = self.fc3(h)  # h: (bs, seq_len, vocab_size)
        return F.log_softmax(h, dim=-1)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)  #z: (bs, hidden_dim)
        return self.decode(z), mu, log_var


def loss_function(recon_x, x, mu, logvar):
    recon_x = recon_x.permute(0, 2, 1)
    # BCE = torch.sum(nn.CrossEntropyLoss(reduction='none')(recon_x, x),
    #                 dim=1)  # torch CrossEntropy为log_softmax与nlllOSS结合
    BCE = torch.sum(nn.NLLLoss(reduction='none')(recon_x, x),
                    dim=1)  # torch CrossEntropy为log_softmax与nlllOSS结合
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return torch.mean(BCE + KLD)


def train(dataset, epoch_num=100):

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    emb_dim = 64
    model = VAE(dataset.n * 2,
                len(dataset.char2id),
                emb_dim=emb_dim,
                residual=True)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    def train_one_epoch(epoch_idx):
        total_loss = 0.
        data_bar = tqdm(dataloader)
        data_bar.ncols = 100
        data_bar.desc = f"Epoch {epoch_idx}"
        for data in data_bar:
            data = data.cuda()
            optimizer.zero_grad()
            output, mu, logvar = model(data)
            loss = loss_function(output, data, mu, logvar)
            loss.backward()
            optimizer.step()
            data_bar.set_postfix(loss=f'{loss:.4f}')
            total_loss += loss
        return total_loss / len(dataloader)

    best_loss = 0.
    for epoch in range(epoch_num):
        model.train()
        loss = train_one_epoch(epoch)
        if epoch == 0:
            best_loss = loss
            torch.save(model, 'checkpoint/model.pt')
        else:
            if best_loss > loss:
                tqdm.write(
                    f'loss descrease({best_loss}->{loss}), saving model...')
                best_loss = loss
                torch.save(model, 'checkpoint/model.pt')
        # model.eval()
        # r = torch.randn((1, emb_dim)).cuda()
        # r = model.decode(r)  #r: (1, seq_len=10, vocab_size)
        # r = r[0].argmax(dim=-1)
        # r = r.cpu().numpy()
        # tqdm.write(
        #     f"{''.join([dataset.id2char[i] for i in r[:dataset.n]])}，{''.join([dataset.id2char[i] for i in r[dataset.n:]])}"
        # )
        generate(model, dataset)


def generate(shi_model, dataset):
    emb_dim = 64
    shi_model = shi_model.cuda()
    shi_model.eval()
    r = torch.randn((1, emb_dim)).cuda()
    r = shi_model.decode(r)  #r: (1, seq_len=10, vocab_size)
    r = r[0].argmax(dim=-1)
    r = r.cpu().numpy()
    print(
        f"{''.join([dataset.id2char[i] for i in r[:dataset.n]])}，{''.join([dataset.id2char[i] for i in r[dataset.n:]])}"
    )


if __name__ == '__main__':
    torch.manual_seed(42)
    if os.path.exists('data/shi.txt'):
        dataset = torch.load('data/shi.pth')
    else:
        dataset = ShiDataset('data/shi.txt')
        torch.save(dataset, 'data/shi.pth')
    # train(dataset)
    shi_model = torch.load('checkpoint/model.pt')
    for i in range(10):
        generate(shi_model, dataset)
