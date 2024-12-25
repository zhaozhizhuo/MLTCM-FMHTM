#一直输出冠心病，难以使用该模型：您好！根据您的描述考虑是冠心病的可能性大一些，建议做一下心脏彩超和动态心电图检查明确病情后再对症治疗比较好!


from transformers import AutoModelForCausalLM, AutoTokenizer

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

# 加载数据和和历史记录
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
    history_i = {}
    query = '你需要依据患者的中医四诊信息判断证型，只需要告诉我证型对应的数字即可，患者的中医四诊信息为，患者的中医四诊信息为：{} \n' \
            '你需要输出以下({})选项中合适该患者的证型，只需要输出对应的选项，并且最多只能输出两个选项，不需要包含其他信息： \n'.format(
        his_sen[i], syndrome)
    ids = []
    for his_label in his_labels[i]:
        id = syndrome_id[his_label]
        ids.append(str(id))
    ans = ' '.join(ids)
    # ans = '；'.join(his_labels[i])
    history_i['role'] = query
    history_i['message'] = ans
    history.append(history_i)

tokenizer = AutoTokenizer.from_pretrained("/mnt/baichuan/", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/mnt/baichuan/", device_map="auto", trust_remote_code=True)

model.eval()
pre_label = []
for sentence in tqdm(sentences_dev):
    query = '你需要依据患者的中医四诊信息判断证型，只需要告诉我证型对应的数字即可，患者的中医四诊信息为，患者的中医四诊信息为：{} \n' \
            '你需要输出以下({})选项中合适该患者的证型，只需要输出对应的选项，并且最多只能输出两个选项，不需要包含其他信息： \n'.format(
        sentence, syndrome)
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to('cuda:0')

    pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
