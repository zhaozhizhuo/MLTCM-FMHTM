from openai import OpenAI
import httpx

client = OpenAI(
    base_url="https://api.xty.app/v1",
    api_key="sk-QUhwNzTTalCLCuy169DcE3Dc70B349EeA325D63bE3E48bE2",
    http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
    ),
)

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
for sentence in sentences:
  query = 'This is a multiple choice TCM syndrome differentiation task. \n' \
          'only need to output the corresponding options, not explanation! You need to select one syndrome suitable for the patient among the ten options ({}): \n'\
          'The patients four diagnosis information is described as: {} \n' \
                .format(syndrome,sentence)
  # print(query)
  completion = client.chat.completions.create(
    max_tokens=100,
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "1"},
        {"role": "user", "content": "{}".format(query)}
    ]
  )
  pre_lable.append(completion.choices[0].message.content)
  print(completion.choices[0].message.content)

with open('./openai_dev_pure.json', 'w', encoding='utf-8') as file:
  for i in range(len(pre_lable)):
    file.write(json.dumps({'id': i, 'label': pre_lable[i]}, ensure_ascii=False) + '\n')