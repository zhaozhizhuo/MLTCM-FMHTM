from train_parser import generate_parser
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os
import sklearn.metrics as metrics
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import shutil
import json
import sys
import random
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import time
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import warnings
# from peft import LoraModel, LoraConfig
from transformers import BertTokenizer
from transformers import BertConfig, BertModel, AdamW

from data_utils import Data_Loader
from loss import loss_fn
from metrics import all_metrics
from evaluation import print_metrics
from src.model.bert import compute_kl_loss


parser = generate_parser()
args = parser.parse_args()

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

class ClassifiyZYBERT(nn.Module):
    def __init__(self, Bertmodel_path, Bertmodel_config):
        super(ClassifiyZYBERT, self).__init__()
        self.PreBert = BertModel.from_pretrained(Bertmodel_path, config=Bertmodel_config)
        # for parameter in self.PreBert.parameters():
        #     parameter.requires_grad = False
        self.clssificion = nn.Linear(768, args.n_labels, bias=True)

    def forward(self, batch=None, token_type_ids=None, return_dict=None):
        input_ids = batch[0]
        attention_mask = batch[1]
        y = batch[2]

        input_ids = input_ids.reshape(-1, args.max_length)
        attention_mask = attention_mask.reshape(-1, args.max_length)

        x_student = self.PreBert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 return_dict=return_dict)
        # print(x_student[1].size())
        h_classificy = self.clssificion(x_student[1])


        if args.threshold == None:
            threshold = find_threshold_micro(h_classificy.cpu().detach().numpy(), y.cpu().detach().numpy())
        else:
            threshold = args.threshold

        yhat = h_classificy >= threshold

        return {"yhat_raw": h_classificy, "yhat": yhat, "y": y ,'t_sne_hidden':torch.mean(x_student[0],dim=1)}


model_path = '../chinese_wwm_pytorch'
model_config = BertConfig.from_pretrained('../chinese_wwm_pytorch/config.json')
tokenizer = BertTokenizer.from_pretrained(model_path)
model = ClassifiyZYBERT(Bertmodel_path=model_path, Bertmodel_config=model_config)
model = model.to(args.device)
# gpus = [0, 1]
# model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict, label_fea = Data_Loader(
        args, tokenizer)

lr = 2e-5
optimizer = AdamW(model.parameters(),
                  lr=lr,  # args.learning_rate - default is 5e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8
                  )
total_steps = len(train_dataloader) * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)
epochs = 15
criterions = nn.CrossEntropyLoss()

best_micro_metric = 0
best_epoch = 0

for epoch_i in range(1, epochs + 1):
    model.train()
    model.zero_grad()

    sum_loss = 0

    outputs = []
    t_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for step, batch in t_bar:
        batch = [tensor.to(args.device) for tensor in batch]

        now_res = model(batch=batch, token_type_ids=None, return_dict=False)
        xx = now_res['yhat_raw']
        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})


        # loss = loss_fn(now_res['yhat_raw'], batch[2])
        label = batch[2].float()
        loss = criterions(xx, label)
        total_loss = loss

        sum_loss = total_loss.item()

        t_bar.update(1)  # 更新进度
        t_bar.set_description("kl_loss:{}   drr_loss:{}   label_loss:{}    loss:{}   lr:{}".format(0, 0, 0, loss, lr))  # 更新描述
        t_bar.refresh()  # 立即显示进度条更新结果

        total_loss.backward()
        optimizer.step()
        scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    print(sum_loss/len(train_dataloader))

    yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
    y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()

    # yhat = np.where(yhat_raw > 0, 1, 0)
    yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy().astype(int)

    metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)

    print_metrics(metric, 'chinese_wwm_pytorch-bert Train_Epoch' + str(epoch_i))
    print_metrics(metric, 'chinese_wwm_pytorch-bert Train_Epoch' + str(epoch_i),
                  os.path.join('./main_007', 'chinese_wwm_pytorch-bert_metric_log'))

    model.eval()
    outputs = []
    for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Test'):
        batch = [tensor.to(args.device) for tensor in batch]
        with torch.no_grad():
            now_res = model(batch=batch, token_type_ids=None, return_dict=False)
        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})

    yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
    y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()

    # '''t-SNE'''
    # import numpy as np
    # from sklearn.datasets import load_digits
    # import matplotlib.pyplot as plt
    # from sklearn import manifold
    #
    # '''t-SNE'''
    # t_sne_hidden = torch.cat([output['t_sne_hidden'] for output in outputs])
    # tsne = manifold.TSNE(n_components=2, perplexity=30)
    # X_tsne = tsne.fit_transform(t_sne_hidden)
    #
    # '''嵌入空间可视化'''
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # plt.figure(figsize=(8, 8))
    # for i in range(X_norm.shape[0]):
    #     # print(type(y[i][0]),y[i][0])
    #     # print(type(np.where(y[i] == 1)[0]),np.where(y[i] == 1)[0])
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(np.where(y[i] == 1)[0][0]),
    #              color=plt.cm.Set1(np.where(y[i] == 1)[0][0]),
    #              fontdict={'weight': 'bold', 'size': 7})
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    # yhat = np.where(yhat_raw > 0, 1, 0)
    yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy().astype(int)

    metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)

    print_metrics(metric, 'chinese_wwm_pytorch-bert Dev_Epoch' + str(epoch_i))
    print_metrics(metric, 'chinese_wwm_pytorch-bert Dev_Epoch' + str(epoch_i),
                  os.path.join('./main_007', 'chinese_wwm_pytorch-bert_metric_log'))

    model.eval()
    outputs = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader),desc='Test'):
        batch = [tensor.to(args.device) for tensor in batch]
        with torch.no_grad():
            now_res = model(batch=batch, token_type_ids=None, return_dict=False)
        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})

    yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
    y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()

    # yhat = np.where(yhat_raw > 0, 1, 0)
    yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy().astype(int)

    metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)

    micro_metric = metric['f1_macro']

    print_metrics(metric, 'chinese_wwm_pytorch-bert Test_Epoch' + str(epoch_i))
    print_metrics(metric, 'chinese_wwm_pytorch-bert Test_Epoch' + str(epoch_i),
                  os.path.join('./main_007', 'chinese_wwm_pytorch-bert_metric_log'))

    if micro_metric > best_micro_metric:
        torch.save(model,
                   '../save/model_best_Chinese_WWM_{}.pkl'.format(args.syndrome_diag))
        best_micro_metric = micro_metric
        best_epoch = epoch_i
print(best_epoch)
# def test_best_model(model):
#     model = torch.load('../save/model_best_Chinese_WWM_{}.pkl'.format(args.syndrome_diag))
#
#     model.eval()
#     outputs = []
#     for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Test'):
#         batch = [tensor.to(args.device) for tensor in batch]
#         with torch.no_grad():
#             now_res = model(batch=batch, token_type_ids=None, return_dict=False)
#         outputs.append({key: value.cpu().detach() for key, value in now_res.items()})
#
#     yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
#     y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()
#
#     '''t-SNE'''
#     import numpy as np
#     from sklearn.datasets import load_digits
#     import matplotlib.pyplot as plt
#     from sklearn import manifold
#     '''t-SNE'''
#     t_sne_hidden = torch.cat([output['t_sne_hidden'] for output in outputs])
#     tsne = manifold.TSNE(n_components=2, perplexity=30)
#     X_tsne = tsne.fit_transform(t_sne_hidden)
#
#     '''嵌入空间可视化'''
#     x_min, x_max = X_tsne.min(0), X_tsne.max(0)
#     X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
#     plt.figure(figsize=(8, 8))
#     for i in range(X_norm.shape[0]):
#         # print(type(y[i][0]),y[i][0])
#         # print(type(np.where(y[i] == 1)[0]),np.where(y[i] == 1)[0])
#         plt.text(X_norm[i, 0], X_norm[i, 1], str(np.where(y[i] == 1)[0][0]),
#                  color=plt.cm.Set1(np.where(y[i] == 1)[0][0]),
#                  fontdict={'weight': 'bold', 'size': 7})
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
#
#     yhat = np.where(yhat_raw > 0, 1, 0)
#     metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)
#
#     print_metrics(metric, 'chinese_wwm_pytorch-bert_BEST Dev_Epoch')
#
#     model.eval()
#     outputs = []
#     for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Test'):
#         batch = [tensor.to(args.device) for tensor in batch]
#         with torch.no_grad():
#             now_res = model(batch=batch, token_type_ids=None, return_dict=False)
#         outputs.append({key: value.cpu().detach() for key, value in now_res.items()})
#
#     yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
#     y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()
#     '''t-SNE'''
#     t_sne_hidden = torch.cat([output['t_sne_hidden'] for output in outputs])
#     tsne = manifold.TSNE(n_components=2, perplexity=30)
#     X_tsne = tsne.fit_transform(t_sne_hidden)
#
#     '''嵌入空间可视化'''
#     x_min, x_max = X_tsne.min(0), X_tsne.max(0)
#     X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
#     plt.figure(figsize=(8, 8))
#     for i in range(X_norm.shape[0]):
#         # print(type(y[i][0]),y[i][0])
#         # print(type(np.where(y[i] == 1)[0]),np.where(y[i] == 1)[0])
#         plt.text(X_norm[i, 0], X_norm[i, 1], str(np.where(y[i] == 1)[0][0]),
#                  color=plt.cm.Set1(np.where(y[i] == 1)[0][0]),
#                  fontdict={'weight': 'bold', 'size': 7})
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
#     yhat = np.where(yhat_raw > 0, 1, 0)
#     metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)
#
#     print_metrics(metric, 'TCM-bert Test_Epoch_BEST')
#
#
# test_best_model(model)