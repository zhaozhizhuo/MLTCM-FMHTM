from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "/mnt/Qwen2-7B-Instruct/",
    torch_dtype="auto",
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained("/mnt/Qwen2-7B-Instruct/")

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
            sentence = content[10] + content[16] + content[12] + content[8] + content[11]
            # if args.syndrome_diag == 'diag':
            # sentence = content[9] + '[SEP]' + content[11] + '[SEP]' + content[13]
            sentences.append(sentence)
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

sentences_test, labels_test = data('./7分类/test.json')
sentences_dev, labels_dev = data('./7分类/dev.json')
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
    history_i['message'] = ans
    history.append(history_i)

pre_lable = []
for sentence in tqdm(sentences_dev):
    query = '你需要依据患者的中医信息判断证型，患者的中医四诊信息为，患者的中医四诊信息为：{} \n' \
            '你需要输出以下({})选项中合适该患者的证型，只需要输出对应的证型名称选项，并且最多只能输出两个选项，不需要包含其他信息： \n'.format(
        sentence, syndrome)

    messages = [
        {"role": "system", "content": "你是一个中医证型辨证专家"+str(history[2]['role'])},
        {"role": "assistant", "content": str(history[2]['message'])},
        {"role": "system", "content": "你是一个中医证型辨证专家"+str(history[1]['role'])},
        {"role": "assistant", "content": str(history[1]['message'])},
        {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].replace('\n','')
    if response == '':
        response = '0'
    pre_lable.append(response)
    print(response)

with open('./qianwen2-7b-dev.json','w',encoding='utf-8') as file:
    for i in range(len(pre_lable)):
        file.write(json.dumps({'id': i, 'label': pre_lable[i]}, ensure_ascii=False) + '\n')


pre_lable = []
for sentence in tqdm(sentences_test):
    query = '你需要依据患者的中医四诊信息判断证型，患者的中医四诊信息为，患者的中医四诊信息为：{} \n' \
            '你需要输出以下({})选项中合适该患者的证型，只需要输出对应的证型名称选项，并且最多只能输出两个选项，不需要包含其他信息： \n'.format(
        sentence, syndrome)

    messages = [
        {"role": "system", "content": "你是一个中医证型辨证专家"+str(history[2]['role'])},
        {"role": "assistant", "content": str(history[2]['message'])},
        {"role": "system", "content": "你是一个中医证型辨证专家"+str(history[1]['role'])},
        {"role": "assistant", "content": str(history[1]['message'])},
        {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    if response == '':
        response = '0'
    pre_lable.append(response)
    print(response)

with open('./qianwen2-7b-test.json','w',encoding='utf-8') as file:
    for i in range(len(pre_lable)):
        file.write(json.dumps({'id': i, 'label': pre_lable[i]}, ensure_ascii=False) + '\n')