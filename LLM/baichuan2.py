#/mnt/baichuan2/

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("/mnt/baichuan2/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/mnt/baichuan2/", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("/mnt/baichuan2/")
# messages = []
# messages.append({"role": "user", "content": "解释一下“温故而知新”"})



import json
from tqdm import tqdm
def data(path):
    sentences = []
    labels = []
    with open(path,'r',encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            content = data
            # if args.syndrome_diag == 'syndrome':
            sentence = content[10] + content[16]
            # if args.syndrome_diag == 'diag':
            # sentence = content[9] + '[SEP]' + content[11] + '[SEP]' + content[13]
            sentences.append(sentence[0:500])
            labels.append(content[-1].split('|'))
    return sentences, labels

#加载数据和和历史记录
letters = [i for i in range(1, 147)]
syndrome_id = {}
with open('./7分类/cardiovascular.json', 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        syndrome_id[letters[i]] = line.replace('\n', '')

syndrome = '  '.join([f"{key}: {value}" for key, value in syndrome_id.items()])
syndrome_id = {value: key for key, value in syndrome_id.items()}

sentences_test, labels = data('./7分类/test.json')
sentences_dev, labels = data('./7分类/dev.json')
his_sen, his_labels = data('./7分类/history.json')

# 历史
history = []
for i in range(len(his_sen)):
    history_i ={}
    query = '你需要依据患者的中医四诊信息判断证型，只需要告诉我证型对应的数字即可，患者的中医四诊信息为，患者的中医四诊信息为：{} \n' \
            '你需要输出以下({})选项中合适该患者的证型，只需要输出对应的选项，并且最多只能输出两个选项，不需要包含其他信息： \n'.format(
        his_sen[i], syndrome)
    ids = []
    for his_label in his_labels[i]:
        id = syndrome_id[his_label]
        ids.append(str(id))
    # ans = ' '.join(ids)
    ans = '；'.join(his_labels[i])
    history_i['role'] = query
    history_i['content'] = ans
    history.append(history_i)



pre_label = []
for sentence in tqdm(sentences_dev):
    query = '你需要依据患者的中医四诊信息判断证型，患者的中医四诊信息为，患者的中医四诊信息为：{} \n' \
            '你需要输出以下({})选项中合适该患者的证型，只需要输出对应的证型名称选项，并且最多只能输出两个选项，不需要包含其他信息： \n'.format(
        sentence, syndrome)

    messages = [
        {"role": "你是一个中医证型辨证专家", "content": str(history[0])},
        {"role": "你是一个中医证型辨证专家", "content": query}
    ]
    response = model.chat(tokenizer, messages)
    # print('history:',history)
    print(response)
    pre_label.append(response)

with open('./baichuan2-dev.json','w',encoding='utf-8') as file:
    for i in range(len(pre_label)):
        file.write(json.dumps({'id': i, 'label': pre_label[i]}, ensure_ascii=False) + '\n')

pre_label = []
for sentence in tqdm(sentences_test):
    query = '你需要依据患者的中医四诊信息判断证型，患者的中医四诊信息为，患者的中医四诊信息为：{} \n' \
            '你需要输出以下({})选项中合适该患者的证型，只需要输出对应的证型名称选项，并且最多只能输出两个选项，不需要包含其他信息： \n'.format(
        sentence, syndrome)

    messages = [
        {"role": "你是一个中医证型辨证专家", "content": str(history[0])},
        {"role": "你是一个中医证型辨证专家", "content": query}
    ]
    response = model.chat(tokenizer, messages)
    print(response)
    pre_label.append(response)

with open('./baichuan2-test.json','w',encoding='utf-8') as file:
    for i in range(len(pre_label)):
        file.write(json.dumps({'id': i, 'label': pre_label[i]}, ensure_ascii=False) + '\n')

# response = model.chat(tokenizer, messages)
# print(response)
