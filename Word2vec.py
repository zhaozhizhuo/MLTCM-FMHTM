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
NUM_EPOCHS = 5
VOCAB_SIZE = 30000
BATCH_SIZE = 256
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100

LOG_FILE = "word-embedding.log"


# def word_tokenize(text):
#     return text.split()

datasets = ['train','dev','test']
Text = []
for dataset in datasets:
    if args.syndrome_diag == 'syndrome':
        path = '../data_preprocess/syndrome_diag/{}.json'.format(dataset)
    with open(path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            data = json.loads(line)
            text = ''.join(data[1:-1])
            for i in text:
                Text.append(i)

# with open("text8.train.txt", "r") as file:
#     text = file.read()  # 一次性读入文件所有内容为一个字符串

# text = [w for w in word_tokenize(text.lower())]

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


# def evaluate(filename, embedding_weights):
#     # embedding_weights是训练之后的embedding向量。
#     if filename.endswith(".csv"):
#         data = pd.read_csv(filename, sep=",")
#     else:
#         data = pd.read_csv(filename, sep="\t")
#     human_similarity = []
#     model_similarity = []
#
#     for i in data.iloc[:, 0:2].index:
#         word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
#         if word1 not in word_to_idx or word2 not in word_to_idx:
#             continue
#         else:
#             word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
#             word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
#             # 在分别取出这两个单词对应的embedding向量，具体为啥是这种取出方式[[word1_idx]]，可以自行研究
#             model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
#             # 用余弦相似度计算这两个100维向量的相似度。这个是模型算出来的相似度
#             human_similarity.append(float(data.iloc[i, 2]))
#             # 这个是人类统计得到的相似度
#
#     return scipy.stats.spearmanr(human_similarity, model_similarity)  # , model_similarity
#     # 因为相似度是浮点数，不是0 1 这些固定标签值，所以不能用准确度评估指标
#     # scipy.stats.spearmanr网址：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
#     # scipy.stats.spearmanr评估两个分布的相似度，有两个返回值correlation, pvalue
#     # correlation是评估相关性的指标（-1，1），越接近1越相关，pvalue值大家可以自己搜索理解


for e in range(NUM_EPOCHS):
    t_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (input_labels, pos_labels, neg_labels) in t_bar:

        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        # model返回的是一个batch所有样本的损失，需要求个平均

        loss.backward()
        optimizer.step()

        t_bar.update(1)  # 更新进度
        t_bar.set_description(
            "epoch:{}   iteration:{}  loss:{}".format(e, i, loss.item()))  # 更新描述
        t_bar.refresh()  # 立即显示进度条更新结果

        if i % 100 == 0:
            with open(LOG_FILE, "a") as fout:
                fout.write("epoch: {}, iter: {}, loss: {}\n".format(e, i, loss.item()))

    embedding_weights = model.input_embeddings()  # 调用最终训练好的embeding词向量
    np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)  # 保存参数
    torch.save(model.state_dict(), "./embedding-bilstm1-{}.th".format(EMBEDDING_SIZE))  # 保存参数


# # ## 在 MEN 和 Simplex-999 数据集上做评估
embedding_weights = model.input_embeddings()
x = embedding_weights
id = word_to_idx['我']
y = torch.tensor(embedding_weights[id])
print(y)


