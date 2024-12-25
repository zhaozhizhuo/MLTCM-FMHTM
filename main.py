import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


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

from src.model.bert import ClassifiyModule
from data_utils import Data_Loader, TCM_SD_Data_Loader
from loss import loss_fn
from metrics import all_metrics
from evaluation import print_metrics
from src.model.bert import compute_kl_loss
# 忽略所有警告
warnings.simplefilter("ignore")

# 设定随机种子值，以确保输出是确定的
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=4):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def run(args):
    # model_path = '../bert-base-uncased'
    # model_config = BertConfig.from_pretrained('../bert-base-uncased/config.json')
    # model_path = '../TCM-bert'
    # model_config = BertConfig.from_pretrained('../TCM-bert/config.json')
    # model_path = '../chinese_wwm_pytorch'
    # model_config = BertConfig.from_pretrained('../chinese_wwm_pytorch/bert_config.json')
    # model_path = '../chinese_roberta_wwm_ext_pytorch'
    # model_config = BertConfig.from_pretrained('../chinese_roberta_wwm_ext_pytorch/config.json')

    model_path = '../{}'.format(args.model_path)
    model_config = BertConfig.from_pretrained('../{}/config.json'.format(args.model_path))
    tokenizer = BertTokenizer.from_pretrained(model_path)  # 加载bert tokenizer
    model = ClassifiyModule(Bertmodel_path=model_path, Bertmodel_config=model_config)
    model = model.to(args.device)
    gpus = args.gpus
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    if args.dataset == 'tcm_sd':
        train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict, label_fea = TCM_SD_Data_Loader(
            args, tokenizer)
    else:
        train_dataloader, test_dataloader, val_dataloader, id2syndrome_dict, label_fea = Data_Loader(
            args, tokenizer)
    if args.model == 'bert':
        hidden_model = 1024
    if args.model == 'longformer':
        hidden_model = 768
    original_tensor = label_fea.reshape(1, args.n_labels, args.define_length, hidden_model)
    label_fea = original_tensor.repeat(len(gpus), 1, 1, 1)

    lr = args.lr
    optimizer = AdamW(model.parameters(),
                      lr=lr,  # args.learning_rate - default is 5e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8
                      )
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    epochs = args.epochs

    criterions = nn.CrossEntropyLoss()
    aw1 = AutomaticWeightedLoss(4)

    micro_metric = 0
    best_micro_metric = 0
    best_macro_metric = 0
    best_epoch = 0

    for epoch_i in range(1, epochs + 1):
        model.train()
        model.zero_grad()

        sum_loss = 0
        sum_num = 1

        outputs = []
        t_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in t_bar:
            batch = [tensor.to(args.device) for tensor in batch]
            label_fea = label_fea.to(args.device)

            # randam label mask site
            label_mask = random.randint(0,args.n_labels-1)
            label_mask_batch = torch.zeros(args.batch_size, args.n_labels)
            label_mask_batch[:,label_mask] = 1
            label_mask_batch = label_mask_batch.to(args.device)

            now_res, distances = model(batch=batch, token_type_ids=None, label=label_fea, label_mask=label_mask, return_dict=False, train='True')
            xx = now_res['yhat_raw']
            outputs.append({key: value.cpu().detach() for key, value in now_res.items()})

            # #仅仅使用分类层
            # kl_loss = 0
            # label_loss = 0
            # distances = 0

            # # 使用HAT -5e-4
            distances = distances.sum() * -1e-3

            # 使用共享注意力
            kl_loss = compute_kl_loss(now_res['yhat_raw'], now_res['yhat_raw_teacher'])
            label_loss = loss_fn(now_res['disl_label_pred'], label_mask_batch.to(now_res['disl_label_pred'].device)) * 10


            label = batch[2].float()
            # loss = loss_fn(xx,label) * 10
            loss = criterions(xx, label.to(xx.device))
            # total_loss = aw1(loss, kl_loss, distances, label_loss)
            total_loss = loss + kl_loss + distances + label_loss

            sum_num += 1
            sum_loss = total_loss.item()

            t_bar.update(1)  # 更新进度
            t_bar.set_description("kl_loss:{} drr_loss:{} label_loss:{} loss:{} total_loss{}".format(kl_loss, distances, label_loss, loss, total_loss))  # 更新描述
            t_bar.refresh()  # 立即显示进度条更新结果

            total_loss.backward()
            # distances.backward(retain_graph=True)

            optimizer.step()
            # scheduler.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        print(sum_loss/len(train_dataloader))

        yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
        y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()

        # max_indices = np.argmax(yhat_raw, axis=1)
        # max_indices = np.array(max_indices, dtype=np.int64)
        # # 创建一个与原始张量相同形状和数据类型的全零张量
        # result = np.zeros_like(yhat_raw, dtype=np.float64)
        # # 将每行最大值的索引处设为 1
        # result[np.arange(len(train_dataloader)*args.batch_size), max_indices] = 1
        # yhat_raw = result

        # yhat = np.where(yhat_raw > 0, 1, 0)
        yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy().astype(int)
        metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)

        print_metrics(metric, 'Train_Epoch' + str(epoch_i))
        print_metrics(metric, 'Train_Epoch' + str(epoch_i),
                      os.path.join('./main_007', 'metric_log'))

        model.eval()
        outputs = []
        for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Test'):
            batch = [tensor.to(args.device) for tensor in batch]
            label_fea = label_fea.to(args.device)
            with torch.no_grad():
                now_res, distances = model(batch=batch, token_type_ids=None, label=label_fea, return_dict=False)
            outputs.append({key: value.cpu().detach() for key, value in now_res.items()})

        yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
        y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()

        # max_indices = np.argmax(yhat_raw, axis=1)
        # max_indices = np.array(max_indices, dtype=np.int64)
        # # 创建一个与原始张量相同形状和数据类型的全零张量
        # result = np.zeros_like(yhat_raw, dtype=np.float64)
        # # 将每行最大值的索引处设为 1
        # result[np.arange(len(val_dataloader) * args.batch_size), max_indices] = 1
        # yhat = result

        # yhat = np.where(yhat_raw > 0, 1, 0)
        yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy().astype(int)
        metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)

        print_metrics(metric, 'Dev_Epoch' + str(epoch_i))
        print_metrics(metric, 'Dev_Epoch' + str(epoch_i),
                      os.path.join('./main_007', 'metric_log'))

        model.eval()
        outputs = []
        for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader),desc='Test'):
            batch = [tensor.to(args.device) for tensor in batch]
            label_fea = label_fea.to(args.device)
            with torch.no_grad():
                now_res, distances = model(batch=batch, token_type_ids=None, label=label_fea, return_dict=False)
            outputs.append({key: value.cpu().detach() for key, value in now_res.items()})

        yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
        y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()

        # max_indices = np.argmax(yhat_raw, axis=1)
        # max_indices = np.array(max_indices, dtype=np.int64)
        # # 创建一个与原始张量相同形状和数据类型的全零张量
        # result = np.zeros_like(yhat_raw, dtype=np.float64)
        # # 将每行最大值的索引处设为 1
        # result[np.arange(len(train_dataloader) * args.batch_size), max_indices] = 1
        # yhat_raw = result

        # yhat = np.where(yhat_raw > 0, 1, 0)
        yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy().astype(int)
        metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)

        macro_metric = metric['f1_macro']
        micro_metric = metric['f1_micro']

        print_metrics(metric, 'Test_Epoch' + str(epoch_i))
        print_metrics(metric, 'Test_Epoch' + str(epoch_i),
                      os.path.join('./main_007', 'metric_log'))

        #11cc--146
        if macro_metric >= best_macro_metric:
            torch.save(model,
                       '{}/model_best_text_{}_{}.pkl'.format(args.data_path,args.model_path, args.syndrome_diag))
            best_micro_metric = micro_metric
            best_macro_metric = macro_metric
            best_epoch = epoch_i
    print('best_epoch:{}'.format(best_epoch))
    test_best_model(model, val_dataloader,test_dataloader,args,label_fea)

def test_best_model(model,val_dataloader,test_dataloader,args,label_fea):
    model = torch.load('{}/model_best_text_{}_{}.pkl'.format(args.data_path,args.model_path, args.syndrome_diag))

    model.eval()
    outputs = []
    for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Test'):
        batch = [tensor.to(args.device) for tensor in batch]
        with torch.no_grad():
            now_res, distances = model(batch=batch, token_type_ids=None, label=label_fea, return_dict=False)
        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})

    yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
    y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()

    # yhat = np.where(yhat_raw > 0, 1, 0)
    yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy().astype(int)

    metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)

    print_metrics(metric, 'BEST Dev_Epoch {}_{}'.format(args.model_path, args.syndrome_diag))

    model.eval()
    outputs = []
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Test'):
        batch = [tensor.to(args.device) for tensor in batch]
        with torch.no_grad():
            now_res, distances = model(batch=batch, token_type_ids=None, label=label_fea, return_dict=False)
        outputs.append({key: value.cpu().detach() for key, value in now_res.items()})

    yhat_raw = torch.cat([output['yhat_raw'] for output in outputs]).cpu().detach().numpy()
    y = torch.cat([output['y'] for output in outputs]).cpu().detach().numpy()

    # yhat = np.where(yhat_raw > 0, 1, 0)
    yhat = torch.cat([output['yhat'] for output in outputs]).cpu().detach().numpy().astype(int)

    metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat_raw)

    print_metrics(metric, 'BEST Test_Epoch {}_{}'.format(args.model_path, args.syndrome_diag))
def main():
    parser = generate_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()