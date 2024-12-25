from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter  # 参数更新和优化函数
from collections import Counter
import numpy as np
import random
import math
# import pandas as pd
import scipy  #
import json
from tqdm import tqdm
import sklearn
from sklearn.metrics.pairwise import cosine_similarity  # 余弦相似度函数

# ###  负例采样就是Skip-Gram模型的输出不是周围词的概率了，是正例和负例的概率

from train_parser import generate_parser
parser = generate_parser()
args = parser.parse_args()


USE_CUDA = torch.cuda.is_available()

random.seed(43)
np.random.seed(43)
torch.manual_seed(43)
if USE_CUDA:
    torch.cuda.manual_seed(43)

K = 10  # 负样本随机采样数量
C = 5  # 周围单词的数量
NUM_EPOCHS = 2
VOCAB_SIZE = 30000
BATCH_SIZE = 256
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100

LOG_FILE = "word-embedding.log"


datasets = ['train','dev','test']
Text = []
for dataset in datasets:
    with open('../data_preprocess/syndrome_diag/{}.json'.format(dataset),'r',encoding='utf-8') as file:
        for line in tqdm(file):
            data = json.loads(line)
            text = ''.join(data[10] + '[SEP]' + data[16]).replace('，', '').replace('。', '').replace('、', '').replace('；', '').replace('：',
                                                                                                            '').replace(
                '？', '').replace('！', '').replace('\t', '').replace('主诉', '').replace('现病史', '').replace('*', '').replace(' ', '').replace('“', '').replace('表格<诊断>内容','').replace('\n', '').replace('（', '').replace('）', '').replace('.', '').replace('”', '').replace('/', '')
            for i in text:
                Text.append(i)

text = Text

vocab = dict(Counter(text).most_common(VOCAB_SIZE - 1))
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))

idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3. / 4.)
word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling


# 实现Dataloader
class Dataset(tud.Dataset):  # 继承tud.Dataset父类

    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(Dataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE - 1) for t in text]
        # get()返回指定键的值，没有则返回默认值
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        # 变成tensor类型，这里变成longtensor，也可以torch.LongTensor

        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)  # 所有单词的总数

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''
        center_word = self.text_encoded[idx]

        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        # replacement=True有放回的取
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], replacement=True)

        return center_word, pos_words, neg_words


dataset = Dataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(next(iter(dataloader))[0].shape)  # 中间词维度data
print(next(iter(dataloader))[1].shape)  # 周围词维度
print(next(iter(dataloader))[2].shape)  # 负样本维度


# ### 定义PyTorch模型
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size  # 30000
        self.embed_size = embed_size  # 100

        # 模型输入，输出是两个一样的矩阵参数nn.Embedding(30000, 100)
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)

        # 权重初始化的一种方法
        initrange = 0.5 / self.embed_size
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围出现过的单词 [batch_size * (c * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (c * 2 * K)]
        return: loss, [batch_size]
        '''
        batch_size = input_labels.size(0)
        input_embedding = self.in_embed(input_labels)  # B * embed_size
        pos_embedding = self.out_embed(pos_labels)  # B * (2C) * embed_size
        neg_embedding = self.out_embed(neg_labels)  # B * (2*C*K) * embed_size

        # torch.bmm()为batch间的矩阵相乘（b,n.m)*(b,m,p)=(b,n,p)
        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()  # B * (2*C)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()  # B * (2*C*K)

        # 下面loss计算就是论文里的公式
        log_pos = F.logsigmoid(log_pos).sum(1)  # batch_size
        log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size
        loss = log_pos + log_neg  # 正样本损失和负样本损失和尽量最大
        return -loss

        # 模型训练有两个矩阵，self.in_embed和self.out_embed两个, 作者认为输入矩阵比较好，舍弃了输出矩阵

    # 取出输入矩阵参数
    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
if USE_CUDA:
    model = model.cuda()

# # ## 在 MEN 和 Simplex-999 数据集上做评估
embedding_weights = model.input_embeddings()

def dataload(embedding_weights, word_to_idx, idx_to_word,path):
    # 找到所有的标签
    syndromes = []
    with open('{}/cardiovascular.json'.format(args.data_path), 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每行内容为JSON对象，并添加到列表中
            syndromes.append(line.replace('\n', ''))

    id2syndrome_dict = {}
    syndrome2id_dict = {}

    y = 0
    for i in range(len(syndromes)):
        id2syndrome_dict[i] = syndromes[i]
        syndrome2id_dict[syndromes[i]] = i

    data_embedding = []
    labels = []
    with open(path,'r',encoding='utf-8') as file:
        for line in tqdm(file):
            data = json.loads(line)
            text = ''.join(data[9] + data[11] + data[13]).replace('，', '').replace('。', '').replace('、', '').replace('；', '').replace('：',
                                                                                                            '').replace(
                '？', '').replace('！', '').replace('\t', '').replace('主诉', '').replace('现病史', '').replace('*', '').replace(' ', '').replace('“', '').replace('表格<诊断>内容','').replace('\n', '').replace('（', '').replace('）', '').replace('.', '').replace('”', '').replace('/', '')
            ids = []
            for i in text[0:512]:
                if i not in word_to_idx:
                    id = word_to_idx["<unk>"]
                else:
                    id = word_to_idx[i]
                ids.append(id)
            if len(ids) < 512:
                ids.extend([word_to_idx["<unk>"]] * (512 - len(ids)))
            tensor_ids = torch.tensor(ids).reshape(1,-1)
            data_embedding.append(tensor_ids)

            labele_sentence = [0] * args.n_labels
            for label in data[-1].split('|'):
                id = syndrome2id_dict[label]
                labele_sentence[id] = 1
            labels.append(torch.tensor(labele_sentence))
    labels = torch.stack(labels, dim=0)
    y = torch.cat(data_embedding, dim=0)
    return y,labels

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

datasets = ['train','dev','test']
for dataset in datasets:
    path = '{}/{}.json'.format(args.data_path,dataset)
    data_ids,labels = dataload(embedding_weights, word_to_idx, idx_to_word, path)
    train_dataset = TensorDataset(data_ids,labels)
    if dataset == 'train':
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,  # 训练样本
            sampler=RandomSampler(train_dataset),  # 随机小批量
            batch_size=32,  # 以小批量进行训练
            drop_last=True,
        )
    if dataset == 'test':
        test_dataloader = torch.utils.data.DataLoader(
            train_dataset,  # 训练样本
            sampler=SequentialSampler(train_dataset),  # 随机小批量
            batch_size=32,  # 以小批量进行训练
            drop_last=True,
        )
    if dataset == 'dev':
        dev_dataloader = torch.utils.data.DataLoader(
            train_dataset,  # 训练样本
            sampler=SequentialSampler(train_dataset),  # 随机小批量
            batch_size=32,  # 以小批量进行训练
            drop_last=True,
        )

def find_threshold_micro(dev_yhat_raw, dev_y):
    dev_yhat_raw_1 = dev_yhat_raw.reshape(-1)
    dev_y_1 = dev_y.reshape(-1)
    sort_arg = np.argsort(dev_yhat_raw_1)
    sort_label = np.take_along_axis(dev_y_1, sort_arg, axis=0)
    label_count = np.sum(sort_label)
    correct = label_count - np.cumsum(sort_label)
    predict = dev_y_1.shape[0] + 1 - np.cumsum(np.ones_like(sort_label))
    f1 = 2 * correct / (predict + label_count)
    sort_yhat_raw = np.take_along_axis(dev_yhat_raw_1, sort_arg, axis=0)
    f1_argmax = np.argmax(f1)
    threshold = sort_yhat_raw[f1_argmax]
    return threshold

import copy
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BILSTM(nn.Module):
    def __init__(self):
        super(BILSTM, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_weights), requires_grad=False)
        self.output_size = EMBEDDING_SIZE

        self.gru = nn.GRU(100, 512, 2, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.2)
        self.classificion = nn.Linear(1024, args.n_labels)

    def init_hidden(self,
                    batch_size: int = 1) -> Variable:
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        h = Variable(torch.zeros(2 * 2, batch_size, 512)).to('cuda:0')
        c = Variable(torch.zeros(2 * 2, batch_size, 512)).to('cuda:0')
        return h, c
    def forward(self,batch_data):
        y = batch_data[1]
        batch_data = batch_data[0]
        batch_size = batch_data.size()[0]
        embeds = self.embedding(batch_data)
        hidden = self.init_hidden(batch_size)

        h0 = torch.zeros(2 * 2, batch_size, 512).to('cuda:0')
        rnn_output, hidden = self.gru(embeds, h0)
        # rnn_output = pad_packed_sequence(rnn_output)[0]
        # rnn_output = rnn_output.permute(1, 0, 2)

        rnn_output = self.dropout(rnn_output)
        pred = self.classificion(rnn_output.max(1)[0])
        if args.threshold == None:
            threshold = find_threshold_micro(pred.cpu().detach().numpy(), y.cpu().detach().numpy())
        else:
            threshold = args.threshold
        yhat = pred >= threshold
        return {"yhat_raw": pred, "yhat": yhat, 'y':y}

from transformers import BertConfig, BertModel, AdamW
from loss import loss_fn
from metrics import all_metrics
from evaluation import print_metrics
import os

model1 = BILSTM()
model1.to('cuda:0')
epochs = 30

optimizer = AdamW(model1.parameters(),
                  lr=5e-6,  # args.learning_rate - default is 5e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8
                  )
criterions = nn.CrossEntropyLoss()

for epoch_i in range(1, epochs + 1):
    model1.train()
    model1.zero_grad()
    outputs = []
    t_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for step, batch in t_bar:
        batch = [tensor.to('cuda:0') for tensor in batch]
        now_res = model1(batch)
        xx = now_res['yhat_raw']
        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})

        label = batch[1].float()
        loss = criterions(xx, label)

        t_bar.update(1)  # 更新进度
        t_bar.set_description(
            "loss:{}".format(loss))  # 更新描述
        t_bar.refresh()  # 立即显示进度条更新结果
        loss.backward()
        optimizer.step()

    yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
    y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()
    # yhat = np.where(yhat_raw > 0, 1, 0)
    yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy().astype(int)

    metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)

    print_metrics(metric, 'Train_Epoch' + str(epoch_i))
    print_metrics(metric, 'Train_Epoch' + str(epoch_i),
                  os.path.join('./main_007', 'metric_log_bilstm'))

    model1.eval()
    outputs = []
    for step, batch in tqdm(enumerate(dev_dataloader), total=len(dev_dataloader), desc='dev'):
        batch = [tensor.to('cuda:0') for tensor in batch]
        with torch.no_grad():
            now_res = model1(batch)

        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})


    yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
    y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()
    # yhat = np.where(yhat_raw > 0, 1, 0)
    yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy().astype(int)

    metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)

    print_metrics(metric, 'dev_Epoch' + str(epoch_i))
    print_metrics(metric, 'dev_Epoch' + str(epoch_i),
                  os.path.join('./main_007', 'metric_log_bilstm'))

    model1.eval()
    outputs = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='test'):
        batch = [tensor.to('cuda:0') for tensor in batch]
        with torch.no_grad():
            now_res = model1(batch)

        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})

    yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
    y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()
    # yhat = np.where(yhat_raw > 0, 1, 0)
    yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy().astype(int)

    metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)

    print_metrics(metric, 'test_Epoch' + str(epoch_i))
    print_metrics(metric, 'test_Epoch' + str(epoch_i),
                  os.path.join('./main_007', 'metric_log_bilstm'))
