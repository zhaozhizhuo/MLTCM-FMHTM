import json
import random
import re
from tqdm import tqdm
import torch
import numpy as np

# 设定随机种子值，以确保输出是确定的
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# choose = ['A','B','C','D','E','F','G']
pre_lables = []
letters = [i for i in range(1,147)]
syndrome_id = {}
with open('./7分类/cardiovascular.json','r',encoding='utf-8') as file:
    for i,line in enumerate(file):
        syndrome_id[letters[i]] = line.replace('\n','')

# syndrome = '  '.join([f"{key}: {value}" for key, value in syndrome_id.items()])
choose = letters

id_syndrome = {v: k for k, v in syndrome_id.items()}
syndrome = list(syndrome_id.keys())
syndrome_name = list(syndrome_id.values())

with open('./qianwen2-7b-dev.json','r',encoding='utf-8') as file:
    for line in tqdm(file):
        pre_lable = []
        data = json.loads(line)
        if data['label'] == None:
            data['label'] = ''

        #按照名称进行选择---qianwen,chatglm,baichuan2
        for line_syndrome in syndrome_name:
            if line_syndrome in data['label']:
                pre_lable.append(id_syndrome[line_syndrome])

        # #使用数字进行匹配--huatuo,llama
        # labele = data['label'].replace('The suitable syndrome for the patient is','').replace("Based on the patient's symptoms and TCM syndrome differentiation, the suitable option would be",'')
        # #筛出空格等，只保留选项
        # matches = re.findall(r'\b\d+\b', labele)
        # if matches:
        #     number = int(matches[0])
        # else:
        #     pre_lable = [random.choice(syndrome)]

        #最原始的匹配方法
        # for i in labele:
        #     if i in choose:
        #         pre_lable.append(id_syndrome[i])

        if len(pre_lable) == 0:
            pre_lable = [random.choice(syndrome)]
        pre_lable = list(set(pre_lable))
        pre_lables.append(pre_lable)


true_labels = []
with open('./7分类/dev.json','r',encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        true_labels.append(data[-1].split('|'))

syndrome2id = {value: key for key, value in syndrome_id.items()}
# syndrome2id = {'肝阳上亢证': '1', '痰湿痹阻证': '2', '气滞血瘀证': '3', '湿热蕴结证': '4', '痰瘀互结证': '5',
#                    '阴虚阳亢证': '6', '气虚血瘀证': '7','痰热蕴结证':'8','心气虚血瘀证':'9','气阴两虚证':'10'}


pre_lables_ids = []
for i in pre_lables:
    pre_lables_id = [0] * len(id_syndrome)
    for j in i:
        j = int(j)
        pre_lables_id[j-int(1)] = 1
    pre_lables_ids.append(pre_lables_id)

true_labels_ids = []
for i in true_labels:
    true_labels_id = [0] * len(id_syndrome)
    for j in i:
        true_labels_id[int(id_syndrome[j])-1] = 1
    true_labels_ids.append(true_labels_id)

import numpy as np
from metrics import all_metrics
from evaluations import print_metrics
yhat = np.array(pre_lables_ids)
y = np.array(true_labels_ids)

metric, ACC = all_metrics(yhat=yhat, y=y, yhat_raw=yhat)
print_metrics(metric)
print(ACC)
