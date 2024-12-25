import json
import torch
import os
from train_parser import generate_parser
from tqdm import tqdm
parser = generate_parser()
args = parser.parse_args()

from simcse import SimCSE
model = SimCSE("../simcse")
# model.to('cuda')
syndromes_text = {}
with open('../data_preprocess/all_know.json', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        name = data['Name']
        text = data['Definition'] + '[SEP]' + data['Typical_performance'] + '[SEP]' + data['Common_isease']
        syndromes_text[name] = text

#找到所有的标签
syndromes = []
with open('../data_preprocess/cardiovascular.json', 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每行内容为JSON对象，并添加到列表中
        syndromes.append(line.replace('\n', ''))


list_synfrome_text = {}
for i in range(len(syndromes)):
    list_synfrome_text[syndromes[i]] = syndromes_text[syndromes[i]].replace('[SEP]', '')[0:args.define_length]

s_a = list(list_synfrome_text.values())


i = 0
sum = 0
num = 0
score_list = []
for i in s_a:
    sentences_a = [i]
    sentences_b = s_a
    similarities = model.similarity(sentences_a, sentences_b)
    print(similarities)
    similarities = similarities.tolist()[0]
    score_list.append(similarities)
    for j in similarities:
        sum = sum + j
        num = num + 1
print(score_list)

print('证型定义之间的平均相似度{}'.format(sum/num))

# 用于存储加载的张量的列表
all_tensors = []
# 逐个加载文件中的张量并添加到列表中
for file_path in range(7):
    loaded_tensor = torch.load('./def_vec/syndrome/sim/label_feature{}.pt'.format(file_path))
    all_tensors.append(loaded_tensor)
label_feature = torch.stack(all_tensors, dim=0).to(args.device)
class_tensors = label_feature.max(1)[0]

from scipy.spatial.distance import cosine
# 计算每个类别的均值向量
# class_means = torch.sum(label_feature,dim=0)

import torch.nn.functional as F

tran = class_tensors
# for i in tran:
#     for j in tran:
#         xx = 1 - F.cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))
#         print('相似度为{}'.format(F.cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))))

def upper_triangle_average(matrix):
    total = 0
    count = 0

    # 遍历行
    for i in range(len(matrix)):
        # 遍历列
        for j in range(len(matrix[i])):
            # 只取上三角部分的元素
            if j >= i:
                total += matrix[i][j]
                count += 1

    # 计算平均值
    if count == 0:
        return 0  # 避免除以零
    else:
        return total / count

sim_i = []
for i in tran:
    sim_j = []
    for j in tran:
        sim_j.append(F.cosine_similarity(i, j, dim=0).item())
    print('class 相似度为{}'.format(sim_j))
    sim_i.append(sim_j)

average = upper_triangle_average(sim_i)
print("上三角元素的平均值为:", average)


# s = 0
# for yy in y:
#     s = s + yy
#
# # print(s)
# # print("类间方差：", class_variance)
# print("总的类间方差：", s/7)