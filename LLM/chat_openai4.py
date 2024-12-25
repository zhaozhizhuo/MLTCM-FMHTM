from openai import OpenAI
import httpx

import os
# 设置 OPENAI_API_KEY 环境变量
os.environ["OPENAI_API_KEY"] = "sk-xyLgQLQdNBSO18p4C4024d4484D345AfA6BbC3Bb950c1eB7"
# 设置 OPENAI_BASE_URL 环境变量
os.environ["OPENAI_BASE_URL"] = "https://api.xiaoai.plus/v1"

client = OpenAI()

import json
def data(path):
    sentences = []
    labels = []
    with open(path,'r',encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            content = data
            sentence = content[10] + content[16]
            sentences.append(sentence[0:500])
            labels.append(content[-1].split('|'))
    return sentences, labels

# syndrome = 'A：心气虚血瘀证   B:痰瘀互结证   C:气滞血瘀证   D:气虚血瘀证   E:肝阳上亢证   F:气阴两虚证   G:痰湿痹阻证'
# syndrome_id = {'心气虚血瘀证': 'A', '痰瘀互结证': 'B', '气滞血瘀证': 'C', '气虚血瘀证': 'D', '肝阳上亢证': 'E',
#                '气阴两虚证': 'F', '痰湿痹阻证': 'G'}
letters = [i for i in range(1,47)]
syndrome_id = {}
with open('./7分类/pure/cardiovascular.json','r',encoding='utf-8') as file:
    for i,line in enumerate(file):
        syndrome_id[letters[i]] = line.replace('\n','')

syndrome = '  '.join([f"{key}: {value}" for key, value in syndrome_id.items()])

# syndrome = 'A：肝阳上亢证   B:痰湿痹阻证   C:气滞血瘀证   D:湿热蕴结证   E:痰瘀互结证   F:阴虚阳亢证   G:气虚血瘀证  H：痰热蕴结证  I:心气虚血瘀证  J：气阴两虚证'
# syndrome_id = {'肝阳上亢证': 'A', '痰湿痹阻证': 'B', '气滞血瘀证': 'C', '湿热蕴结证': 'D', '痰瘀互结证': 'E',
#                '阴虚阳亢证': 'F', '气虚血瘀证': 'G','痰热蕴结证':'H','心气虚血瘀证':'I','气阴两虚证':'J'}
sentences, labels = data('./7分类/pure/dev.json')


pre_lable = []
for i,sentence in enumerate(sentences):
  query = 'This is a multiple choice TCM syndrome differentiation task. \n' \
          'only need to output the corresponding options, not explanation! You need to select one syndrome suitable for the patient among the ten options ({}): \n'\
          'The patients four diagnosis information is described as: {} \n' \
                .format(syndrome,sentence)
  # print(query)
  completion = client.chat.completions.create(
    max_tokens=50,
    model="gpt-4-0125-preview",
    messages=[
        {"role": "system", "content": "1"},
        {"role": "user", "content": "{}".format(query)}
    ]
  )
  pre_lable.append(completion.choices[0].message.content)
  print('第{}条数据:'.format(i),completion.choices[0].message.content)

with open('./openai4_pure_dev.json', 'w', encoding='utf-8') as file:
  for i in range(len(pre_lable)):
    file.write(json.dumps({'id': i, 'label': pre_lable[i]}, ensure_ascii=False) + '\n')